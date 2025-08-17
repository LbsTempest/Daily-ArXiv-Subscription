from src.workflow_runner import WorkflowRunner

if __name__ == "__main__":
    runner = WorkflowRunner(config_path="config.yml")
    runner.run()