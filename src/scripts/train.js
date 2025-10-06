// scripts/train.js
import { tensor2d, util, sequential, layers, train } from '@tensorflow/tfjs-node';
import { readFileSync, existsSync, mkdirSync, writeFileSync } from 'fs';
import { resolve, join } from 'path';

const DATA_PATH = resolve(__dirname, '../public/exoplanets.json');

function loadData() {
  const raw = readFileSync(DATA_PATH, 'utf8');
  return JSON.parse(raw);
}

// Configura aquí las features que usarás
const FEATURE_KEYS = ['dec','density','duration','epoch','period','ra','radius']; // ajustable
const LABEL_KEY = 'isExoplanet';

// Trata 0 en campos donde 0 significa NA (ajusta según tus datos)
function isMissingValue(key, val) {
  // ejemplo: radius, density, temperature => 0 es missing
  const zeroMissingKeys = ['radius','density','temperature','stellarRad','stellarTemp'];
  return zeroMissingKeys.includes(key) && (val === 0 || val === null || val === undefined);
}

function computeImputeValues(data, featureKeys) {
  // mediana por feature (más robusta ante outliers)
  const medians = {};
  featureKeys.forEach(k => {
    const vals = data
      .map(d => d[k])
      .filter(v => v !== null && v !== undefined && !(isMissingValue(k, v)));
    vals.sort((a,b) => a-b);
    if (vals.length === 0) medians[k] = 0;
    else {
      const mid = Math.floor(vals.length/2);
      medians[k] = (vals.length % 2 === 0) ? (vals[mid-1]+vals[mid])/2 : vals[mid];
    }
  });
  return medians;
}

function imputeAndExtract(data, featureKeys, medians) {
  const X = [];
  const y = [];
  data.forEach(d => {
    const row = featureKeys.map(k => {
      let v = d[k];
      if (v === null || v === undefined || isMissingValue(k, v)) {
        v = medians[k];
      }
      // si aún NaN, poner 0
      if (Number.isNaN(v) || v === null || v === undefined) v = 0;
      return Number(v);
    });
    X.push(row);
    y.push(Number(d[LABEL_KEY] || 0));
  });
  return { X, y };
}

function computeStandardScaler(X) {
  const nFeatures = X[0].length;
  const mean = Array(nFeatures).fill(0);
  const std = Array(nFeatures).fill(0);
  const N = X.length;

  for (let j=0;j<nFeatures;j++) {
    let s = 0;
    for (let i=0;i<N;i++) s += X[i][j];
    mean[j] = s / N;
  }
  for (let j=0;j<nFeatures;j++) {
    let s = 0;
    for (let i=0;i<N;i++) s += Math.pow(X[i][j] - mean[j], 2);
    std[j] = Math.sqrt(s / N) || 1;
  }
  const Xs = X.map(row => row.map((v,j) => (v - mean[j]) / std[j]));
  return { Xs, mean, std };
}

async function main() {
  const data = loadData();
  console.log('Loaded rows:', data.length);

  // 1. compute medians and impute
  const medians = computeImputeValues(data, FEATURE_KEYS);
  const { X, y } = imputeAndExtract(data, FEATURE_KEYS, medians);

  // 2. scale (z-score)
  const { Xs, mean, std } = computeStandardScaler(X);

  const xs = tensor2d(Xs);
  const ys = tensor2d(y, [y.length, 1]);

  // 3. split train/test
  const total = xs.shape[0];
  const testSize = Math.floor(total * 0.15);
  const trainSize = total - testSize;

  // Shuffle
  const indices = util.createShuffledIndices(total);
  const trainIdx = indices.slice(0, trainSize);
  const testIdx = indices.slice(trainSize);

  const xTrain = xs.gather(trainIdx);
  const yTrain = ys.gather(trainIdx);
  const xTest = xs.gather(testIdx);
  const yTest = ys.gather(testIdx);

  // 4. model
  const model = sequential();
  model.add(layers.dense({ inputShape: [FEATURE_KEYS.length], units: 32, activation: 'relu' }));
  model.add(layers.dropout({rate: 0.2}));
  model.add(layers.dense({ units: 16, activation: 'relu' }));
  model.add(layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({
    optimizer: train.adam(0.001),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });

  console.log('Training...');
  await model.fit(xTrain, yTrain, {
    epochs: 50,
    batchSize: 64,
    validationData: [xTest, yTest],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch ${epoch+1} loss=${logs.loss.toFixed(4)} val_loss=${(logs.val_loss||0).toFixed(4)} acc=${(logs.acc||logs.acc).toFixed(3)}`);
      }
    }
  });

  // 5. evaluar
  const evalRes = model.evaluate(xTest, yTest);
  const loss = (await evalRes[0].data())[0];
  const acc = (await evalRes[1].data())[0];
  console.log('Test loss:', loss, 'Test acc:', acc);

  // 6. guardar modelo y metadata (mean/std/medians)
  const outDir = resolve(__dirname, '../public/model');
  if (!existsSync(outDir)) mkdirSync(outDir, { recursive: true });

  // Guardar modelo en formato filesystem
  await model.save(`file://${outDir}/tfmodel`);
  console.log('Model saved to', outDir);

  // Guardar scaler y medians para usar en cliente/predicción
  const meta = { FEATURE_KEYS, mean, std, medians };
  writeFileSync(join(outDir, 'meta.json'), JSON.stringify(meta, null, 2));
  console.log('Saved meta.json');
  process.exit(0);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
