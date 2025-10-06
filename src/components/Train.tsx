"use client";
import React, { useEffect, useState } from "react";
import { trainModel, loadModel, predictExoplanet } from "@/lib/exoplanetModel";
import exoplanetData from "@/data/exoplanets.json";

export default function ExoplanetPredictor() {
  const [model, setModel] = useState<any>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState({ epoch: 0, loss: 0, accuracy: 0 });
  const [prediction, setPrediction] = useState(null);
  const [isReady, setIsReady] = useState(false);
  const [isReadyPrediction, setIsReadyPrediction] = useState(false);

  // Campos del formulario
  const [inputData, setInputData] = useState({
    period: "",
    transitDepth: "",
    duration: "",
    epoch: "",
    radius: "",
    density: "",
    temperature: "",
    stellarRad: "",
    stellarTemp: "",
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputData({ ...inputData, [e.target.name]: e.target.value });
  };

  const handleTrain = async () => {
    setIsTraining(true);
    const m = await trainModel(exoplanetData, setProgress);
    setModel(m);
    setIsTraining(false);
  };

  const handlePredict = async () => {
    const m = model || (await loadModel());
    if (!m) return alert("FIRST train or load the model!");

    // Convertir valores del formulario a números
    const numericData = Object.fromEntries(
      Object.entries(inputData).map(([k, v]) => [k, parseFloat(v)])
    );

    const result = await predictExoplanet(m, numericData);
    setIsReadyPrediction(true);
    setPrediction(result);
  };

  useEffect(() => {
    // Revisar si el modelo está guardado
    const modelInfo = localStorage.getItem("tensorflowjs_models/exoplanet-model/info");
    if (modelInfo) {
      loadModel().then((m) => {
        setModel(m);
        setIsReady(true);
      });
    }
  }, []);

  return (
    <div className="space-y-6 w-full text-white">

      {/* train */}
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
                <div><strong>Epoch:</strong> {progress.epoch}/150</div>
                <div><strong>Loss:</strong> {progress.loss}</div>
                <div><strong>Accuracy:</strong> {progress.accuracy}%</div>
            </div>
        </div>

      {/* form */}
      {isReady && <form
        onSubmit={(e) => {
          e.preventDefault();
          handlePredict();
        }}
        className="bg-gray-900 p-6 rounded-xl shadow-md space-y-4"
      >
        <h2 className="text-xl font-semibold text-white">🔢 Input Data</h2>
        <div className="grid grid-cols-2 gap-4 text-left">
          {Object.keys(inputData).map((key) => (
            <div key={key} className="flex flex-col">
              <label htmlFor={key} className="text-gray-300 capitalize">
                {key}
              </label>
              <input
                id={key}
                name={key}
                type="number"
                step="any"
                value={inputData[key as keyof typeof inputData]}
                onChange={handleChange}
                className="p-2 rounded-md bg-gray-700 text-white focus:ring-2 focus:ring-cyan-400"
                required
              />
            </div>
          ))}
        </div>

        <button
          type="submit"
          disabled={isTraining}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 rounded-lg mt-4"
        >
          🔮 Predict
        </button>
      </form>}

      {/* Resultado */}
      {isReady  &&  isReadyPrediction &&(
        <p
          className={`text-2xl font-semibold mt-4 ${
            prediction ? "text-green-400" : "text-red-400"
          }`}
        >
          {prediction
            ? "🪐 This object is likely an Exoplanet!"
            : "🌌 This object is probably not an Exoplanet."}
        </p>
      )}
    </div>
  );
}
