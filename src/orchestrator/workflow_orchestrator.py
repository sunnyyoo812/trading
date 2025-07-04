class WorkflowOrchestrator:
    
    def training():
        df = create_latest_historical_data()
        model_trainer = ModelTrainer()
        model = model_trainer.train_model(df)
        return model

        