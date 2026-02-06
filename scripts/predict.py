import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import hydra
from agents.mdoc_agent import MDocAgent
from scripts.retrieve import RetrievedDataset
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    # GPU Settings
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.mdoc_agent.cuda_visible_devices
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

    # Load configurations
    for agent_config in cfg.mdoc_agent.agents:
        agent_name = agent_config.agent
        model_name = agent_config.model
        agent_cfg = hydra.compose(config_name="agent/"+agent_name, overrides=[]).agent
        model_cfg = hydra.compose(config_name="model/"+model_name, overrides=[]).model
        agent_config.agent = agent_cfg
        agent_config.model = model_cfg
    
    cfg.mdoc_agent.sum_agent.agent = hydra.compose(config_name="agent/"+cfg.mdoc_agent.sum_agent.agent, overrides=[]).agent
    cfg.mdoc_agent.sum_agent.model = hydra.compose(config_name="model/"+cfg.mdoc_agent.sum_agent.model, overrides=[]).model

    # LlamaIndex Retriever
    embed_model = HuggingFaceEmbedding(model_name=cfg.retrieval.embed_model)
    Settings.embed_model = embed_model
    Settings.llm = None

    storage = StorageContext.from_defaults(persist_dir=cfg.retrieval.persist_dir)
    index = load_index_from_storage(storage)

    retriever = index.as_retriever(
        similarity_top_k=cfg.retrieval.top_k
    )
    
    dataset = RetrievedDataset(
        cfg.dataset.input_path,
        retriever=retriever,
        top_k=cfg.retrieval.top_k
    )
    
    # Multi-agent processing
    mdoc_agent = MDocAgent(cfg.mdoc_agent)
    samples = mdoc_agent.predict_dataset(dataset)
    mdoc_agent.dump_results(samples, cfg.dataset.output_path)
    print(f"Saved -> {cfg.dataset.output_path}")
    
if __name__ == "__main__":
    main()