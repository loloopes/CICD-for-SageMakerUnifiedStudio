"""Test that the ML deployment created a live endpoint and a batch transform job."""
import os
import boto3
from datetime import datetime, timedelta, timezone


def _client():
    region = os.environ.get("DOMAIN_REGION", "us-east-1")
    return region, boto3.client("sagemaker", region_name=region)


def _recent_cutoff():
    return datetime.now(timezone.utc) - timedelta(hours=2)


def test_endpoint_deployed():
    """Verify that a SageMaker inference endpoint was created and is in service."""
    region, sagemaker_client = _client()

    response = sagemaker_client.list_endpoints(
        NameContains="realistic-classifier",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=10,
    )
    if not response["Endpoints"]:
        response = sagemaker_client.list_endpoints(
            NameContains="sagemaker-scikit-learn",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=10,
        )

    assert response["Endpoints"], (
        "No SageMaker endpoints found with pattern 'realistic-classifier' "
        "or 'sagemaker-scikit-learn'"
    )

    latest_endpoint = response["Endpoints"][0]
    endpoint_name = latest_endpoint["EndpointName"]
    endpoint_details = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)

    endpoint_status = endpoint_details["EndpointStatus"]
    creation_time = endpoint_details["CreationTime"]

    print(f"✓ Endpoint found: {endpoint_name}")
    print(f"✓ Status: {endpoint_status}")
    print(f"✓ Created: {creation_time}")
    print(f"✓ Region: {region}")
    print(f"✓ Endpoint ARN: {endpoint_details['EndpointArn']}")

    assert creation_time > _recent_cutoff(), f"Endpoint is too old: {creation_time}"
    assert endpoint_status == "InService", f"Endpoint status is {endpoint_status}, expected InService"

    endpoint_config_name = endpoint_details["EndpointConfigName"]
    config_details = sagemaker_client.describe_endpoint_config(
        EndpointConfigName=endpoint_config_name
    )

    print(f"✓ Endpoint config: {endpoint_config_name}")
    print(f"✓ Production variants: {len(config_details['ProductionVariants'])}")

    assert config_details["ProductionVariants"], "No production variants configured"

    variant = config_details["ProductionVariants"][0]
    print(f"✓ Instance type: {variant['InstanceType']}")
    print(f"✓ Initial instance count: {variant['InitialInstanceCount']}")
    print(f"✓ Model name: {variant['ModelName']}")

    model_details = sagemaker_client.describe_model(ModelName=variant["ModelName"])
    print(f"✓ Model ARN: {model_details['ModelArn']}")
    print(f"✓ Execution role: {model_details['ExecutionRoleArn']}")

    print("\n✅ Endpoint deployment verified successfully")


def test_batch_transform_completed():
    """Verify that a SageMaker batch transform job completed successfully."""
    region, sagemaker_client = _client()

    response = sagemaker_client.list_transform_jobs(
        NameContains="realistic-classifier",
        StatusEquals="Completed",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=10,
    )

    summaries = response.get("TransformJobSummaries", [])
    assert summaries, (
        "No completed SageMaker batch transform jobs found "
        "with pattern 'realistic-classifier'"
    )

    latest = summaries[0]
    job_name = latest["TransformJobName"]
    details = sagemaker_client.describe_transform_job(TransformJobName=job_name)

    status = details["TransformJobStatus"]
    creation_time = details["CreationTime"]

    print(f"✓ Transform job found: {job_name}")
    print(f"✓ Status: {status}")
    print(f"✓ Created: {creation_time}")
    print(f"✓ Region: {region}")
    print(f"✓ Transform job ARN: {details['TransformJobArn']}")

    assert creation_time > _recent_cutoff(), f"Transform job is too old: {creation_time}"
    assert status == "Completed", f"Transform job status is {status}, expected Completed"

    resources = details["TransformResources"]
    print(f"✓ Instance type: {resources['InstanceType']}")
    print(f"✓ Instance count: {resources['InstanceCount']}")
    print(f"✓ Model name: {details['ModelName']}")
    print(f"✓ Output path: {details['TransformOutput']['S3OutputPath']}")

    model_details = sagemaker_client.describe_model(ModelName=details["ModelName"])
    print(f"✓ Model ARN: {model_details['ModelArn']}")
    print(f"✓ Execution role: {model_details['ExecutionRoleArn']}")

    print("\n✅ Batch transform verified successfully")
