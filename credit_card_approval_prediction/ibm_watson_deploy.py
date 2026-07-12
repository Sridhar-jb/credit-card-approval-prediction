from __future__ import annotations

import os
from pathlib import Path

from ibm_watson_machine_learning import APIClient

from src.schemas import MODEL_PATH


def main() -> None:
    api_key = os.environ.get("IBM_CLOUD_API_KEY")
    space_id = os.environ.get("IBM_WML_SPACE_ID")
    url = os.environ.get("IBM_WML_URL", "https://us-south.ml.cloud.ibm.com")

    if not api_key or not space_id:
        raise SystemExit("Set IBM_CLOUD_API_KEY and IBM_WML_SPACE_ID before deployment.")

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise SystemExit("Model file not found. Run `python train_model.py` first.")

    client = APIClient({"url": url, "apikey": api_key})
    client.set.default_space(space_id)

    software_spec_id = client.software_specifications.get_id_by_name("runtime-22.2-py3.10")
    metadata = {
        client.repository.ModelMetaNames.NAME: "Credit Card Approval Prediction",
        client.repository.ModelMetaNames.TYPE: "scikit-learn_1.1",
        client.repository.ModelMetaNames.SOFTWARE_SPEC_ID: software_spec_id,
    }

    stored_model = client.repository.store_model(model=str(model_path), meta_props=metadata)
    model_id = client.repository.get_model_id(stored_model)

    deployment = client.deployments.create(
        artifact_uid=model_id,
        meta_props={
            client.deployments.ConfigurationMetaNames.NAME: "Credit Card Approval Online Deployment",
            client.deployments.ConfigurationMetaNames.ONLINE: {},
        },
    )
    deployment_id = client.deployments.get_id(deployment)

    print(f"Stored model id: {model_id}")
    print(f"Deployment id: {deployment_id}")


if __name__ == "__main__":
    main()

