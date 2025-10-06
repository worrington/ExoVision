import * as tf from "@tensorflow/tfjs";

// Normalización automática (Min-Max)
function normalizeFeature(arr) {
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  return arr.map(v => (v - min) / (max - min || 1));
}

export async function trainModel(data, onProgress) {
  const fields = [
    "period", "transitDepth", "duration", "epoch",
    "radius", "density", "temperature", "stellarRad", "stellarTemp"
  ];

  const featureArrays = fields.map(f => normalizeFeature(data.map(d => d[f])));
  const features = data.map((_, i) => featureArrays.map(fArr => fArr[i]));
  const labels = data.map(d => d.isExoplanet);

  const xs = tf.tensor2d(features);
  const ys = tf.tensor2d(labels, [labels.length, 1]);

  // Calcular pesos de clase (para dataset desbalanceado)
  const numPos = labels.filter(x => x === 1).length;
  const numNeg = labels.length - numPos;
  const total = labels.length;
  const classWeight = {
    0: total / (2 * numNeg),
    1: total / (2 * numPos),
  };

  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [9], units: 128, activation: "relu" }),
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

  await model.save("localstorage://exoplanet-model");
  xs.dispose();
  ys.dispose();
  console.log("✅ Modelo entrenado con normalización + class weights");
  return model;
}


export async function loadModel() {
  try {
    const model = await tf.loadLayersModel("localstorage://exoplanet-model");
    console.log("📦 Modelo cargado desde localStorage");
    return model;
  } catch {
    console.log("⚠️ No hay modelo guardado, entrena uno nuevo");
    return null;
  }
}

// Predice si un nuevo dato es un exoplaneta
export async function predictExoplanet(model, newData) {
  const input = tf.tensor2d([[
    newData.dec / 100,
    newData.density / 10,
    newData.duration / 10,
    newData.epoch / 1000,
    newData.period / 1000,
    newData.ra / 360
  ]]);

  const prediction = model.predict(input);
  const result = (await prediction.data())[0];

  return result > 0.5 ? "Exoplaneta probable 🌍" : "No exoplaneta ❌";
}