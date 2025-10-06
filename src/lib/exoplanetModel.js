import * as tf from "@tensorflow/tfjs";

// Normaliza un valor usando min-max
function normalizeValue(value, min, max) {
  return (value - min) / (max - min || 1);
}

// Entrena el modelo
export async function trainModel(data, onProgress) {
  const fields = [
    "period", "transitDepth", "duration", "epoch",
    "radius", "density", "temperature", "stellarRad", "stellarTemp"
  ];

  // Calcular min y max de cada feature
  const featureStats = {};
  fields.forEach(f => {
    const arr = data.map(d => d[f]);
    featureStats[f] = { min: Math.min(...arr), max: Math.max(...arr) };
  });

  // Normalizar features
  const features = data.map(d =>
    fields.map(f => normalizeValue(d[f], featureStats[f].min, featureStats[f].max))
  );
  const labels = data.map(d => d.isExoplanet);

  const xs = tf.tensor2d(features);
  const ys = tf.tensor2d(labels, [labels.length, 1]);

  // Pesos de clase
  const numPos = labels.filter(x => x === 1).length;
  const numNeg = labels.length - numPos;
  const total = labels.length;
  const classWeight = {
    0: total / (2 * numNeg),
    1: total / (2 * numPos),
  };

  // Crear modelo
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [fields.length], units: 128, activation: "relu" }),
      tf.layers.dropout({ rate: 0.3 }),
      tf.layers.dense({ units: 64, activation: "relu" }),
      tf.layers.dropout({ rate: 0.3 }),
      tf.layers.dense({ units: 32, activation: "relu" }),
      tf.layers.dense({ units: 1, activation: "sigmoid" }),
    ],
  });

  model.compile({
    optimizer: tf.train.adam(0.0005),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  // Entrenar modelo
  await model.fit(xs, ys, {
    epochs: 150,
    batchSize: 64,
    validationSplit: 0.2,
    shuffle: true,
    classWeight,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (onProgress)
          onProgress({
            epoch: epoch + 1,
            loss: logs.loss.toFixed(4),
            accuracy: (logs.acc * 100).toFixed(2),
            valAcc: (logs.val_acc * 100).toFixed(2),
          });
      },
    },
  });

  // Guardar modelo y featureStats
  await model.save("localstorage://exoplanet-model");
  localStorage.setItem("featureStats", JSON.stringify(featureStats));

  xs.dispose();
  ys.dispose();

  return model;
}

// Cargar modelo
export async function loadModel() {
  try {
    const model = await tf.loadLayersModel("localstorage://exoplanet-model");
    console.log("📦 Model loaded from localStorage");
    return model;
  } catch {
    console.log("⚠️ are no model found in localStorage");
    return null;
  }
}

// Predecir nuevo planeta
export async function predictExoplanet(model, planet) {
  const featureStats = JSON.parse(localStorage.getItem("featureStats"));
  if (!featureStats) {
    throw new Error("⚠️ No se encontró featureStats. Entrena primero el modelo.");
  }

  const inputArray = Object.keys(featureStats).map(f =>
    normalizeValue(planet[f], featureStats[f].min, featureStats[f].max)
  );

  const input = tf.tensor2d([inputArray]);
  const prediction = model.predict(input);
  const result = (await prediction.data())[0];

  return result > 0.5 ? true : false;
}
