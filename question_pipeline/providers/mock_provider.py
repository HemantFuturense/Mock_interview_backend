import hashlib
import re
from typing import List, Dict, Any, Optional
from .base import ResearchProvider, LLMProvider, EmbeddingProvider, LLMResult
from ..models import ResearchOutput
from ..cost_tracker import cost_tracker
from .local_provider import LocalEmbeddingProvider

# 20 distinct calibrated questions per role (4 per experience band)
ROLE_BAND_QUESTIONS = {
    "Machine Learning Engineer": {
        "0-1": [
            ("Explain the mathematical difference between L1 and L2 regularization and how each impacts weight sparsity in logistic regression.", ["Python", "Numpy", "Scikit-Learn"]),
            ("How do you handle severe class imbalance in tabular binary classification using stratified sampling and loss weighting?", ["Scikit-Learn", "Metrics", "Imbalance"]),
            ("Write a clean PyTorch module implementing a multi-layer perceptron with dropout and batch normalization layers.", ["PyTorch", "Deep Learning", "Python"]),
            ("What is the purpose of a validation set versus a test set, and how do you prevent data leakage during feature preprocessing?", ["Validation", "Preprocessing", "Cross-Validation"])
        ],
        "1-2": [
            ("How do you vectorize text preprocessing pipelines using TF-IDF and token hashing to minimize memory during batch training?", ["NLP", "Vectorization", "Python"]),
            ("Explain how learning rate warm-up and cosine annealing schedules improve optimization stability in AdamW.", ["Optimization", "AdamW", "PyTorch"]),
            ("How do you serialize and serve an XGBoost model using ONNX Runtime with sub-5ms inference latency?", ["ONNX", "XGBoost", "Model Serving"]),
            ("Describe your strategy for tuning hyperparameters using Bayesian Optimization with Optuna on limited compute budgets.", ["Optuna", "Hyperparameter Tuning", "Bayesian"])
        ],
        "3-5": [
            ("How do you design a low-latency feature store architecture to serve online features with strict sub-10ms p99 SLAs?", ["Redis", "Feature Store", "Feast"]),
            ("Explain how you would detect concept drift and covariate shift in production ML pipelines before business KPIs degrade.", ["Evidently AI", "MLflow", "Drift Detection"]),
            ("How do you architect an asynchronous embedding generation service that batches vector queries dynamically?", ["FastAPI", "AsyncIO", "Vector Embeddings"]),
            ("Compare the throughput and memory footprints of INT8 quantization versus FP16 mixed-precision inference in production.", ["Quantization", "TensorRT", "Precision"])
        ],
        "5-8": [
            ("How do you design a distributed training pipeline across multi-node GPU clusters using PyTorch FSDP and ZeRO-3?", ["FSDP", "Distributed Training", "CUDA"]),
            ("Architect a real-time recommendation system combining two-tower retrieval with a heavy ranking model and re-ranking layer.", ["Two-Tower", "RecSys", "System Design"]),
            ("How do you prevent catastrophic forgetting when fine-tuning foundational models on proprietary enterprise data?", ["LoRA", "PEFT", "Fine-Tuning"]),
            ("Describe your strategy for continuous online evaluation using multi-armed bandits instead of static A/B tests.", ["Bandits", "Reinforcement Learning", "A/B Testing"])
        ],
        "8+": [
            ("Architect the long-term enterprise ML platform roadmap balancing on-prem GPU clusters, cloud elasticity, and multi-tenant billing.", ["ML Platform", "FinOps", "Cluster Management"]),
            ("How do you establish organization-wide model governance, bias auditing, and regulatory compliance standards for LLM systems?", ["AI Governance", "Compliance", "Ethics"]),
            ("Define the end-to-end technical strategy for migrating high-volume ML workloads to custom silicon (e.g. TPUs / Inferentia).", ["Silicon Strategy", "Hardware Acceleration", "TPU"]),
            ("How do you resolve systemic engineering trade-offs between model latency, context length, and infrastructure cost across 50+ services?", ["Trade-offs", "Enterprise Architecture", "Cost Optimization"])
        ]
    },
    "Data Engineer": {
        "0-1": [
            ("Write an optimized SQL query using window functions (ROW_NUMBER and LAG) to identify user session boundaries.", ["SQL", "PostgreSQL", "Window Functions"]),
            ("Explain the internal differences between INNER JOIN, LEFT JOIN, and CROSS JOIN in database execution plans.", ["Database Indexing", "SQL", "Relational Models"]),
            ("How do you build an idempotent Python script that ingests partitioned JSON files from S3 into PostgreSQL?", ["Python", "S3", "Idempotency"]),
            ("What is the difference between star schema and snowflake schema in dimensional data modeling?", ["Data Modeling", "OLAP", "Kimball"])
        ],
        "1-2": [
            ("How do you tune Apache Spark shuffle partitions and memory fractions to avoid OutOfMemory (OOM) errors during large joins?", ["Apache Spark", "PySpark", "Memory Tuning"]),
            ("Explain how columnar file formats like Apache Parquet use dictionary encoding and run-length encoding for compression.", ["Parquet", "Compression", "File Formats"]),
            ("How do you build a robust Airflow DAG with SLAs, retry policies with exponential backoff, and Slack alerting?", ["Airflow", "Orchestration", "Monitoring"]),
            ("Describe how you partition and cluster tables in Google BigQuery to optimize query performance and reduce scan costs.", ["BigQuery", "Cost Optimization", "Partitioning"])
        ],
        "3-5": [
            ("How do you guarantee exactly-once processing semantics in Apache Flink stream processing using Chandy-Lamport checkpointing?", ["Apache Flink", "Checkpointing", "Streaming"]),
            ("Design a Change Data Capture (CDC) pipeline using Debezium and Kafka to synchronize transactional DBs into Apache Iceberg.", ["CDC", "Debezium", "Iceberg"]),
            ("How do you implement schema evolution without breaking downstream consumers in an Apache Kafka event streaming architecture?", ["Schema Registry", "Avro", "Kafka"]),
            ("Compare the trade-offs of Apache Iceberg versus Delta Lake for transactional data lakehouses with concurrent writers.", ["Iceberg", "Delta Lake", "Lakehouse"])
        ],
        "5-8": [
            ("Architect a hybrid real-time and batch Kappa architecture serving sub-second analytics over petabyte-scale event streams.", ["Kappa Architecture", "Real-Time Analytics", "ClickHouse"]),
            ("How do you implement automated data contract verification and circuit breakers at ingestion boundaries across 20+ producer teams?", ["Data Contracts", "Data Quality", "Circuit Breakers"]),
            ("Design a multi-region disaster recovery and failover strategy for critical Apache Kafka and streaming clusters with zero data loss.", ["Kafka DR", "Multi-Region", "Disaster Recovery"]),
            ("How do you optimize data warehouse storage and compute compute credits across enterprise Snowflake instances spending $1M+ annually?", ["Snowflake", "FinOps", "Resource Monitors"])
        ],
        "8+": [
            ("Define the technical vision and roadmap for transitioning a monolithic data warehouse to a decentralized Data Mesh architecture.", ["Data Mesh", "Domain Ownership", "Data Strategy"]),
            ("How do you design an enterprise-wide metadata, data cataloging, and automated data lineage governance platform?", ["Data Lineage", "OpenMetadata", "Governance"]),
            ("Establish data engineering principles for zero-trust data access, field-level encryption, and automated GDPR/CCPA erasure pipelines.", ["Zero-Trust", "Data Security", "Compliance"]),
            ("Architect the corporate data platform foundation supporting autonomous analytics, AI feature stores, and ad-hoc data science exploration.", ["Enterprise Architecture", "Platform Engineering", "Lakehouse"])
        ]
    },
    "MLOps Engineer (Cloud Deployment)": {
        "0-1": [
            ("Write a multi-stage Dockerfile to containerize a FastAPI model serving service minimizing image layer sizes.", ["Docker", "FastAPI", "Containerization"]),
            ("How do you configure GitHub Actions to lint code, run pytest suites, and push versioned images to Amazon ECR?", ["CI/CD", "GitHub Actions", "Docker"]),
            ("Explain the difference between Liveness and Readiness probes in Kubernetes and how they affect model startup.", ["Kubernetes", "Probes", "DevOps"]),
            ("How do you manage sensitive environment variables and API keys securely in Docker Compose and Kubernetes secrets?", ["Secrets Management", "Environment Variables", "Security"])
        ],
        "1-2": [
            ("How do you configure Kubernetes Horizontal Pod Autoscaler (HPA) based on custom Prometheus metrics like request concurrency?", ["Kubernetes", "HPA", "Prometheus"]),
            ("Explain how to use MLflow Model Registry to transition models through Staging, Production, and Archived stages with webhooks.", ["MLflow", "Model Registry", "Versioning"]),
            ("How do you implement automated model artifact validation and smoke testing in an Argo Workflows pipeline?", ["Argo Workflows", "Validation", "Pipelines"]),
            ("Describe how you configure Grafana dashboards to monitor p99 latency, error rates, and GPU utilization for Triton Inference Server.", ["Grafana", "Triton", "Observability"])
        ],
        "3-5": [
            ("Design an automated canary deployment pipeline for ML models with Istio service mesh and automated rollback on error budget burn.", ["Istio", "Canary", "Progressive Delivery"]),
            ("How do you solve cold-start latency spikes for serverless GPU model serving under sudden 10x burst traffic spikes?", ["Cold Starts", "Serverless GPU", "Scale-to-Zero"]),
            ("Architect an automated drift-detection trigger that initiates continuous model retraining and evaluation workflows.", ["Kubeflow", "Continuous Retraining", "Automation"]),
            ("How do you secure ML cloud pipelines with IAM least-privilege roles, VPC endpoints, and KMS customer-managed encryption keys?", ["AWS IAM", "VPC Endpoints", "KMS"])
        ],
        "5-8": [
            ("Architect a multi-cloud, multi-region inference infrastructure supporting 50,000 queries per second with 99.99% availability.", ["Multi-Cloud", "High Availability", "Global Load Balancing"]),
            ("How do you design a shared multi-tenant GPU cluster orchestration platform ensuring fair queuing and preemption across teams?", ["Ray", "Slurm", "GPU Sharing"]),
            ("Design an automated vulnerability patching and zero-downtime base-image upgrade lifecycle for thousands of active model pods.", ["Zero-Downtime", "Vulnerability Scanning", "Container Security"]),
            ("Implement FinOps cost-governance for cloud machine learning pipelines, reducing idle GPU waste via Spot instance automation.", ["FinOps", "Spot Instances", "Cost Optimization"])
        ],
        "8+": [
            ("Define the enterprise MLOps architectural roadmap unifying model development, feature storage, deployment, and real-time monitoring.", ["MLOps Roadmap", "Platform Architecture", "Strategy"]),
            ("How do you structure cross-functional SLA and SLO agreements between ML researchers, platform engineers, and business stakeholders?", ["SLAs/SLOs", "Stakeholder Management", "Engineering Leadership"]),
            ("Architect a sovereign, air-gapped ML deployment platform for defense/healthcare customers with zero outbound internet access.", ["Air-Gapped", "Compliance", "Security Architecture"]),
            ("Lead the organizational transition from manual model releases to fully autonomous GitOps-driven continuous machine learning delivery.", ["GitOps", "Autonomous Delivery", "Culture Transformation"])
        ]
    },
    "Computer Vision Engineer": {
        "0-1": [
            ("Explain the operation of 2D convolutional kernels, stride, and padding in edge detection and feature extraction.", ["CNN", "OpenCV", "Convolution"]),
            ("How do you prevent overfitting when training a ResNet classifier on a small dataset using standard data augmentation?", ["Data Augmentation", "ResNet", "Transfer Learning"]),
            ("Write a PyTorch script to load custom image datasets with multi-worker DataLoader and image normalization transforms.", ["PyTorch", "DataLoader", "Transforms"]),
            ("What is the difference between object classification, semantic segmentation, and instance segmentation?", ["Segmentation", "Object Detection", "Foundations"])
        ],
        "1-2": [
            ("How do you compute Mean Average Precision (mAP@0.5:0.95) for bounding box evaluation across multi-class datasets?", ["mAP", "Object Detection", "Evaluation"]),
            ("Explain the role of Non-Maximum Suppression (NMS) and Soft-NMS in resolving overlapping bounding box predictions.", ["NMS", "YOLO", "Post-Processing"]),
            ("How do you optimize an image inference pipeline using ONNX Runtime and TensorRT FP16 precision on NVIDIA GPUs?", ["TensorRT", "ONNX", "GPU Inference"]),
            ("Describe techniques to handle class imbalance in fine-grained visual classification using Focal Loss.", ["Focal Loss", "Loss Functions", "Imbalance"])
        ],
        "3-5": [
            ("Design an edge computer vision pipeline to run 60 FPS multi-camera object tracking on resource-constrained NVIDIA Jetson devices.", ["DeepStream", "Jetson", "Edge AI"]),
            ("Compare the operational trade-offs of single-stage detectors (YOLOv8/v9) versus two-stage detectors (Faster R-CNN) in production.", ["YOLO", "Faster R-CNN", "Latency Trade-offs"]),
            ("How do you train vision models that remain robust against extreme weather conditions, lens glare, and motion blur?", ["Domain Adaptation", "Robustness", "Perception"]),
            ("Architect a visual search pipeline using deep metric learning embeddings, ArcFace loss, and Milvus vector search index.", ["Metric Learning", "Visual Search", "Milvus"])
        ],
        "5-8": [
            ("Design a real-time multi-modal sensor fusion system combining LiDAR point clouds and camera streams for 3D bounding box estimation.", ["Sensor Fusion", "LiDAR", "Autonomous Vehicles"]),
            ("How do you design a Vision Transformer (ViT) architecture optimized for linear computational complexity in high-resolution medical imaging?", ["Vision Transformers", "Attention Mechanisms", "Medical Imaging"]),
            ("Architect a self-supervised pretraining pipeline (e.g. DINO / MAE) utilizing millions of unlabelled satellite images.", ["Self-Supervised", "DINO", "Satellite Imagery"]),
            ("How do you isolate and mitigate adversarial patch attacks against deployed deep learning vision models in security cameras?", ["Adversarial Robustness", "Security", "Patch Attacks"])
        ],
        "8+": [
            ("Define the end-to-end perception engineering strategy for safety-critical autonomous robotic navigation across diverse environments.", ["Robotics", "Safety-Critical", "System Architecture"]),
            ("Lead the architectural evaluation of on-device neural processing units (NPUs) versus centralized cloud inference for 10M+ smart devices.", ["Edge vs Cloud", "NPU Strategy", "Hardware Evaluation"]),
            ("Establish validation protocols and automated synthetic data generation frameworks using generative diffusion models for rare failure cases.", ["Synthetic Data", "Diffusion Models", "Validation Protocols"]),
            ("Architect the long-term vision technology stack balancing proprietary custom architectures and emerging vision-language foundation models.", ["VLM", "Foundation Models", "Technology Strategy"])
        ]
    },
    "Product Manager (Tech)": {
        "0-1": [
            ("How do you write clear, unambiguous engineering user stories with testable acceptance criteria?", ["Agile", "User Stories", "Acceptance Criteria"]),
            ("Explain the difference between customer outputs (features shipped) and customer outcomes (value delivered).", ["Outcomes vs Outputs", "Product Discovery", "KPIs"]),
            ("How do you use user telemetry data to identify the primary drop-off point in an onboarding funnel?", ["Funnel Analysis", "Analytics", "Mixpanel"]),
            ("Describe how you conduct a competitive feature matrix analysis when evaluating entering a new market.", ["Competitive Analysis", "Market Research", "Benchmarking"])
        ],
        "1-2": [
            ("How do you apply the RICE framework (Reach, Impact, Confidence, Effort) to prioritize backlogs with 50+ competing feature requests?", ["RICE", "Prioritization", "Backlog Management"]),
            ("Describe how you design an A/B testing experiment, determine minimum sample size, and interpret statistical significance.", ["A/B Testing", "Experimentation", "Statistics"]),
            ("How do you define the North Star metric and input metrics for a developer-facing REST/GraphQL API platform?", ["North Star", "Developer Experience", "API Metrics"]),
            ("Walk through a time customer discovery interviews invalidated your initial product hypothesis and how you responded.", ["Customer Discovery", "Validation", "User Research"])
        ],
        "3-5": [
            ("How do you negotiate and balance critical technical debt remediation versus revenue-generating feature requests with engineering leads?", ["Tech Debt", "Engineering Alignment", "Roadmapping"]),
            ("Design a product strategy for transitioning a complex desktop enterprise software to a cloud-native SaaS subscription platform.", ["SaaS Migration", "Product Strategy", "Monetization"]),
            ("How do you design a self-service product onboarding experience that decreases Time-to-Value (TTV) by 40% for technical users?", ["Time to Value", "PLG", "Self-Service"]),
            ("Describe your framework for evaluating build-versus-buy-versus-partner decisions for core enterprise capabilities.", ["Build vs Buy", "Strategic Decision", "Partnerships"])
        ],
        "5-8": [
            ("Lead the product vision and commercialization strategy for an internal data platform transforming into an external revenue product.", ["Productization", "Platform Strategy", "Go-To-Market"]),
            ("How do you manage complex multi-stakeholder trade-offs when launching a breaking API change impacting thousands of external partners?", ["Breaking Changes", "Developer Relations", "Partner Management"]),
            ("Architect an experimentation framework supporting multi-armed bandits and real-time contextual personalization in high-volume e-commerce.", ["Contextual Personalization", "Personalization", "Algorithms"]),
            ("How do you structure product pricing tiers and consumption-based billing models to maximize enterprise expansion revenue?", ["Pricing Strategy", "Consumption Billing", "Enterprise Sales"])
        ],
        "8+": [
            ("Define the multi-year product portfolio strategy and capital allocation across emerging venture bets and mature legacy products.", ["Portfolio Strategy", "Capital Allocation", "Executive Leadership"]),
            ("How do you build and nurture an autonomous product organization driven by discovery, empowered teams, and strong business outcomes?", ["Product Org Design", "Empowered Teams", "Leadership"]),
            ("Lead an enterprise-wide product pivot driven by sudden market disruptions caused by generative AI technologies.", ["Strategic Pivot", "Disruptive Innovation", "Change Management"]),
            ("Establish product governance frameworks that ensure ethical AI principles, data privacy compliance, and customer trust at scale.", ["AI Ethics", "Privacy", "Trust & Safety"])
        ]
    },
    "Senior RTL / Logic Design Engineer": {
        "0-1": [
            ("Write a clean SystemVerilog module implementing a parameterized synchronous FIFO with full and empty flag generation.", ["SystemVerilog", "FIFO", "RTL Design"]),
            ("Explain the difference between blocking (=) and non-blocking (<=) procedural assignments in synthesis versus simulation.", ["Verilog", "Synthesis", "Simulation Semantics"]),
            ("How do you write a finite state machine (FSM) using the two-always-block methodology separating state and output logic?", ["FSM", "State Machines", "Logic Design"]),
            ("What is setup time and hold time in flip-flops, and what causes metastability under asynchronous inputs?", ["Timing", "Metastability", "Digital Design"])
        ],
        "1-2": [
            ("How do you design a dual-flop clock domain crossing (CDC) synchronizer for single-bit control signals?", ["CDC", "Synchronizer", "Clock Domains"]),
            ("Explain techniques for dynamic power reduction at the RTL level using automated architectural clock gating.", ["Clock Gating", "Low Power", "Dynamic Power"]),
            ("How do you write synthesis constraints (SDC) specifying clock definitions, input/output delays, and false paths?", ["SDC", "Synthesis Constraints", "Synopsys"]),
            ("Describe how you resolve setup timing violations on critical data paths using register retiming and logic restructuring.", ["Setup Timing", "Optimization", "Timing Closure"])
        ],
        "3-5": [
            ("Design an asynchronous multi-word FIFO with Gray-coded read and write pointers across asynchronous clock domains.", ["Async FIFO", "Gray Codes", "CDC"]),
            ("How do you implement low-power architectures using multiple power domains, level shifters, and isolation cells (UPF/CPF)?", ["UPF", "Power Domains", "Level Shifters"]),
            ("How do you diagnose and fix hold time violations across slow-slow and fast-fast process voltage temperature (PVT) corners in STA?", ["STA", "PVT Corners", "PrimeTime"]),
            ("Design a high-throughput multi-stage pipelined 64-bit integer arithmetic unit with hazard detection and forwarding logic.", ["Pipelining", "Hazard Detection", "Computer Architecture"])
        ],
        "5-8": [
            ("Architect an AXI4 interconnect crossbar switch supporting out-of-order transaction completions, multiple masters, and burst transfers.", ["AXI4", "Interconnect", "Bus Protocols"]),
            ("How do you implement functional safety features (ISO 26262) including lockstep cores and ECC parity protection for automotive ASICs?", ["ISO 26262", "Functional Safety", "Lockstep"]),
            ("Describe your methodology for closing timing on a multi-GHz SoC targeting 3nm/5nm FinFET technology nodes.", ["FinFET", "Timing Closure", "Advanced Nodes"]),
            ("Design a comprehensive static timing analysis (STA) flow handling on-chip variation (AOCV/POCV) and signal integrity crosstalk.", ["AOCV", "POCV", "Signal Integrity"])
        ],
        "8+": [
            ("Define the microarchitecture specification and physical floorplan strategy for a multi-core heterogeneous AI accelerator SoC.", ["SoC Architecture", "Microarchitecture", "Chip Floorplanning"]),
            ("Lead the cross-disciplinary silicon strategy balancing performance, power, die area (PPA), and foundry fabrication yield constraints.", ["PPA", "Foundry Relations", "Silicon Strategy"]),
            ("Establish organization-wide RTL design standards, formal verification sign-off methodologies, and reusable IP repository governance.", ["IP Reuse", "Formal Verification", "Design Governance"]),
            ("Architect the multi-die chiplet interconnect architecture utilizing UCIe (Universal Chiplet Interconnect Express) high-speed links.", ["Chiplets", "UCIe", "Packaging Strategy"])
        ]
    }
}

class MockResearchProvider(ResearchProvider):
    """Mock research provider used exclusively when MOCK_MODE=true for unit tests."""
    @property
    def provider_name(self) -> str:
        return "mock_research"

    async def research_role(self, role: str) -> ResearchOutput:
        summary = (
            f"Comprehensive technical overview for {role}. "
            "Key architectures: Microservices, Distributed Systems, Event-Driven streaming. "
            "Interview focus: High-load scalability, real-time troubleshooting, latency trade-offs."
        )
        source_hash = hashlib.sha256(summary.encode("utf-8")).hexdigest()
        return ResearchOutput(
            role=role,
            technologies=["Python", "PyTorch", "Docker", "Kubernetes", "PostgreSQL", "Kafka"],
            engineering_practices=["CI/CD", "Automated Testing", "Observability", "Design Patterns"],
            responsibilities=[f"Lead design and architecture for {role}"],
            interview_topics=["Distributed Architecture", "Failure Recovery", "Performance Optimization"],
            trends=["Serverless ML", "Cost optimization", "Modern data architectures"],
            raw_summary=summary,
            source_references=["Mock Technical Journal 2026", "Mock Engineering Whitepaper"],
            source_hash=source_hash,
            actual_provider="mock_research"
        )

    async def research_company(self, company: str) -> ResearchOutput:
        summary = f"Mock technical profile for {company} with cloud infrastructure and microservices."
        return ResearchOutput(
            role=company,
            company=company,
            raw_summary=summary,
            source_references=[f"Mock Tech Profile {company}"],
            source_hash=hashlib.sha256(summary.encode("utf-8")).hexdigest(),
            actual_provider="mock_research"
        )

class MockLLMProvider(LLMProvider):
    """Mock LLM provider used exclusively when MOCK_MODE=true for offline testing."""
    def __init__(self, provider_name: str = "mock_gemini"):
        self._name = provider_name

    @property
    def provider_name(self) -> str:
        return self._name

    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        task_name: str = "mock_generation"
    ) -> LLMResult:
        if "synthesis" in task_name:
            data = {
                "modules": [
                    {"topic": "Architecture & Scalability", "content": "Microservices, distributed state, high throughput."},
                    {"topic": "Troubleshooting & Failover", "content": "Root-cause isolation, circuit breakers, backpressure."},
                    {"topic": "Tooling & Best Practices", "content": "Automated pipelines, observability, test harness."},
                    {"topic": "Trade-offs & Cost Optimization", "content": "Latency vs throughput, cloud infrastructure cost management."}
                ]
            }
        elif "validation" in task_name:
            data = {
                "passed": True,
                "score": 0.88,
                "reasoning": "High-quality question testing realistic architectural and production trade-offs.",
                "uncertain": False
            }
        else:
            # Parse parameters from prompt
            role_match = re.search(r"Role:\s*'([^']+)'", prompt)
            band_match = re.search(r"Experience Band:\s*'([^']+)'", prompt)
            paradigm_match = re.search(r"Question Paradigm:\s*'([^']+)'", prompt)
            scope_match = re.search(r"Target Scope:\s*'([^']+)'", prompt)

            role = role_match.group(1) if role_match else "Machine Learning Engineer"
            band = band_match.group(1) if band_match else "3-5"
            paradigm = paradigm_match.group(1) if paradigm_match else "ARCHITECTURE"
            scope = scope_match.group(1) if scope_match else "UNIVERSAL"

            # Difficulty calibration by band
            diff_map = {"0-1": 4, "1-2": 5, "3-5": 7, "5-8": 8, "8+": 9}
            base_diff = diff_map.get(band, 7)

            # Retrieve templates for role and band
            role_data = ROLE_BAND_QUESTIONS.get(role, ROLE_BAND_QUESTIONS["Machine Learning Engineer"])
            band_templates = role_data.get(band, role_data["3-5"])

            questions = []
            for i, (tmpl_text, skills) in enumerate(band_templates):
                questions.append({
                    "question": tmpl_text,
                    "role": role,
                    "experience_band": band,
                    "difficulty": min(10, base_diff + (i % 2)),
                    "technical_depth": min(10, base_diff + (i % 2)),
                    "problem_complexity": min(10, base_diff),
                    "architecture_complexity": min(10, base_diff + 1 if base_diff >= 6 else base_diff - 1),
                    "troubleshooting": min(10, base_diff),
                    "business_complexity": max(1, base_diff - 2),
                    "decision_making": min(10, base_diff),
                    "leadership_ownership": max(1, base_diff - 3) if band in ("0-1", "1-2") else min(10, base_diff - 1),
                    "question_type": "technical" if i % 2 == 0 else "system_design",
                    "paradigm": paradigm,
                    "mandatory_skills": skills,
                    "scope": scope,
                    "domains": ["CLOUD_SYSTEMS", "ENTERPRISE_PLATFORMS"],
                    "applicable_companies": ["Google", "Amazon", "Microsoft", "Meta", "Apple"]
                })

            data = {"questions": questions}

        cost_tracker.record_usage(
            provider=self._name,
            model="mock-model",
            task=task_name,
            input_tokens=150,
            output_tokens=250
        )

        return LLMResult(
            text=str(data),
            data=data,
            input_tokens=150,
            output_tokens=250,
            model="mock-model",
            provider=self._name
        )

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        task_name: str = "text"
    ) -> LLMResult:
        res = await self.generate_json(prompt, system_prompt, task_name)
        return res

MockEmbeddingProvider = LocalEmbeddingProvider
