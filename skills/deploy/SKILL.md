---
name: deploy
description: Deploy applications, dashboards, and models to various targets including local Docker, Snowflake, Streamlit Cloud, or cloud platforms. Works with any data application.
---

# Deployment

## When to Use
- Deploying trained models or dashboards to production
- Setting up Docker containers, cloud services, or Streamlit apps
- Creating inference APIs

## Supported Targets

- **Local**: Docker containers with docker-compose
- **Snowflake**: Model Registry + Streamlit in Snowflake
- **Streamlit Cloud**: Community Cloud deployment
- **AWS**: ECS, Lambda, SageMaker
- **GCP**: Cloud Run, Vertex AI

## Workflow

1. Validate deployment requirements
2. Package model/application
3. Configure target environment
4. Deploy and verify health endpoints
5. Set up monitoring
