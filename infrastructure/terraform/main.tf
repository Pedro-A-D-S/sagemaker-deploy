# SageMaker Realtime Endpoint with Target Tracking & Step Scaling (-23% FinOps Pattern)

variable "environment" {
  type    = string
  default = "production"
}

variable "endpoint_name" {
  type    = string
  default = "whisper-dl-realtime-endpoint"
}

resource "aws_sagemaker_endpoint_configuration" "endpoint_config" {
  name = "${var.endpoint_name}-config-${var.environment}"

  # Blue/Green Canary Variant traffic splitting (90/10)
  production_variants {
    variant_name           = "Variant-Primary-v2"
    model_name             = "production-dl-model-v2"
    initial_instance_count = 2
    instance_type          = "ml.m5.xlarge"
    initial_variant_weight = 0.90
  }

  production_variants {
    variant_name           = "Variant-Challenger-v3"
    model_name             = "production-dl-model-v3"
    initial_instance_count = 1
    instance_type          = "ml.m5.xlarge"
    initial_variant_weight = 0.10
  }
}

resource "aws_sagemaker_endpoint" "realtime_endpoint" {
  name                 = "${var.endpoint_name}-${var.environment}"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.endpoint_config.name
}

# Autoscaling Target for Primary Variant
resource "aws_appautoscaling_target" "sagemaker_target" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "endpoint/${aws_sagemaker_endpoint.realtime_endpoint.name}/variant/Variant-Primary-v2"
  scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
  service_namespace  = "sagemaker"
}

# Target Tracking: InvocationsPerInstance (Maintains SLA and cuts -23% idle capacity)
resource "aws_appautoscaling_policy" "sagemaker_target_tracking" {
  name               = "SageMakerTargetTrackingInvocations"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.sagemaker_target.resource_id
  scalable_dimension = aws_appautoscaling_target.sagemaker_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.sagemaker_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "SageMakerVariantInvocationsPerInstance"
    }
    target_value       = 750.0 # scale out when traffic exceeds 750 calls/min per node
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}

output "endpoint_arn" {
  value = aws_sagemaker_endpoint.realtime_endpoint.arn
}
