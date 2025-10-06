"use client";
import React, { useState } from "react";
import { trainModel, loadModel, predictExoplanet } from "@/lib/exoplanetModel";
import exoplanetData from "@/data/exoplanets.json";

export default function ExoplanetPredictor() {
  const [model, setModel] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState({ epoch: 0, loss: 0, accuracy: 0 });
  const [prediction, setPrediction] = useState("");

  const handleTrain = async () => {
    setIsTraining(true);
    const m = await trainModel(exoplanetData, setProgress);
    setModel(m);
    setIsTraining(false);
  };

  

  const handlePredict = async () => {
    const m = model || (await loadModel());
    const result = await predictExoplanet(m, {
      period: 9.48803557,
      transitDepth: 615.8,
      duration: 2.9575,
      epoch: 170.53875,
      radius: 2.26,
      density: 0.08663127028471196,
      temperature: 793,
      stellarRad: 0.927,
      stellarTemp: 5455,
    });

    setPrediction(result);
  };

  return (
    <div className="p-6 text-center space-y-4">
      <h1 className="text-2xl font-bold">🔭 Exoplanet AI Predictor</h1>

        <button
            onClick={handleTrain}
            disabled={isTraining}
            style={{
            padding: '15px 30px',
            background: isTraining ? '#666' : '#4caf50',
            color: '#fff',
            border: 'none',
            borderRadius: '8px',
            cursor: isTraining ? 'not-allowed' : 'pointer',
            fontWeight: 'bold',
            fontSize: '16px',
            marginBottom: '20px'
            }}
        >
            {isTraining ? '🔄 Training...' : '🚀 Train Neural Network'}
        </button>

        <div style={{ padding: '15px', background: '#2a2a4e', borderRadius: '8px', marginBottom: '20px' }}>
            <h3 style={{ color: '#4fc3f7', marginBottom: '10px' }}>Training Progress</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '15px' }}>
                <div><strong>Epoch:</strong> {progress.epoch}/50</div>
                <div><strong>Loss:</strong> {progress.loss}</div>
                <div><strong>Accuracy:</strong> {progress.accuracy}%</div>
            </div>
        </div>

        <button
            onClick={handlePredict}
            disabled={isTraining}
            className="px-4 py-2 bg-green-600 text-white rounded-lg"
        >
            Predecir nuevo dato
        </button>

        {prediction && <p className="text-xl font-semibold mt-4">{prediction}</p>}
    </div>
  );
}
