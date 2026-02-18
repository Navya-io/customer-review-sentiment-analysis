"""
score.py  — Azure ML Online Endpoint Scoring Script
----------------------------------------------------
Loaded once at startup; called per-request for real-time inference.
Registers with Azure ML and connects to CRM via REST API.
"""

import os, json, time, logging
import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

log = logging.getLogger(__name__)
LABELS = ["Negative", "Neutral", "Positive"]
model, tokenizer, device = None, None, None


def init():
    """Called once when the endpoint container starts."""
    global model, tokenizer, device
    model_dir = os.getenv("AZUREML_MODEL_DIR", "models/distilbert-sentiment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    log.info(f"Model loaded from {model_dir} on {device}")


def run(raw_data: str) -> str:
    """
    Called per request.
    Accepts:
        {"text": "single review string"}
    or  {"texts": ["review1", "review2", ...]}   (batch, max 32)
    Returns JSON with sentiment, confidence, scores, and latency.
    """
    t0 = time.time()
    data = json.loads(raw_data)

    texts = data.get("texts") or [data["text"]]
    texts = texts[:32]  # hard cap for latency SLA

    enc = tokenizer(
        texts, truncation=True, padding="max_length",
        max_length=128, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(
            input_ids      = enc["input_ids"].to(device),
            attention_mask = enc["attention_mask"].to(device)
        ).logits

    probs  = torch.softmax(logits, dim=1).cpu().numpy()
    preds  = np.argmax(probs, axis=1)
    ms     = round((time.time() - t0) * 1000, 1)

    results = [
        {
            "sentiment":  LABELS[p],
            "confidence": round(float(probs[i][p]), 4),
            "scores": {
                "Negative": round(float(probs[i][0]), 4),
                "Neutral":  round(float(probs[i][1]), 4),
                "Positive": round(float(probs[i][2]), 4),
            },
            "text_preview": texts[i][:80],
        }
        for i, p in enumerate(preds)
    ]

    response = {
        "results":             results,
        "inference_time_ms":   ms,
        "model":               "distilbert-sentiment-v1",
    }

    # Single input → unwrap list for cleaner CRM integration
    if len(results) == 1:
        response.update(results[0])
        del response["results"]

    return json.dumps(response)


# ═══════════════════════════════════════════════════════════════════════════════
# deploy.py — One-shot deployment to Azure ML Managed Online Endpoint
# ═══════════════════════════════════════════════════════════════════════════════
"""
Usage:
    python azure_ml/deploy.py \
        --subscription-id <YOUR_SUB_ID> \
        --resource-group  <YOUR_RG> \
        --workspace       <YOUR_WS> \
        --model-path      models/distilbert-sentiment \
        --endpoint-name   navya-sentiment-ep
"""

import argparse

def deploy(subscription_id, resource_group, workspace, model_path, endpoint_name):
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import (
        ManagedOnlineEndpoint, ManagedOnlineDeployment,
        Model, Environment, CodeConfiguration
    )
    from azure.identity import DefaultAzureCredential

    ml = MLClient(
        DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )

    # ── Register model ─────────────────────────────────────────────────────────
    registered = ml.models.create_or_update(
        Model(name="distilbert-sentiment", path=model_path, type="custom_model")
    )
    print(f"Registered model: {registered.name} v{registered.version}")

    # ── Create endpoint ────────────────────────────────────────────────────────
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Real-time customer review sentiment classifier",
        auth_mode="key",
        tags={"project": "customer-sentiment", "owner": "navya-manjunatha"},
    )
    ml.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Endpoint '{endpoint_name}' created")

    # ── Environment ────────────────────────────────────────────────────────────
    env = Environment(
        name="distilbert-env",
        conda_file="azure_ml/environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    )

    # ── Deployment ─────────────────────────────────────────────────────────────
    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=endpoint_name,
        model=registered,
        environment=env,
        code_configuration=CodeConfiguration(
            code="azure_ml",
            scoring_script="score.py"
        ),
        instance_type="Standard_DS3_v2",
        instance_count=1,
        request_settings={"request_timeout_ms": 5000, "max_concurrent_requests_per_instance": 8},
    )
    ml.online_deployments.begin_create_or_update(deployment).result()

    # Route 100% traffic to blue
    endpoint.traffic = {"blue": 100}
    ml.online_endpoints.begin_create_or_update(endpoint).result()

    # Print scoring URI & key
    ep = ml.online_endpoints.get(endpoint_name)
    key = ml.online_endpoints.get_keys(endpoint_name).primary_key
    print(f"\n✅ Deployment complete!")
    print(f"   Scoring URI : {ep.scoring_uri}")
    print(f"   Primary Key : {key[:8]}…")
    print(f"\nTest with:")
    print(f'   curl -X POST "{ep.scoring_uri}" \\')
    print(f'     -H "Authorization: Bearer {key[:8]}..." \\')
    print(f'     -H "Content-Type: application/json" \\')
    print(f"     -d '{{\"text\": \"The battery life is amazing!\"}}'")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subscription-id", required=True)
    p.add_argument("--resource-group",  required=True)
    p.add_argument("--workspace",       required=True)
    p.add_argument("--model-path",      default="models/distilbert-sentiment")
    p.add_argument("--endpoint-name",   default="navya-sentiment-ep")
    args = p.parse_args()
    deploy(args.subscription_id, args.resource_group,
           args.workspace, args.model_path, args.endpoint_name)
