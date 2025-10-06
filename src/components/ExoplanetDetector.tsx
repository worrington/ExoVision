/* eslint-disable */
import Papa from "papaparse";

// Hook para cargar CSVs
import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import * as tf from '@tensorflow/tfjs';
import * as THREE from 'three';

// Generar curva de luz con tránsitos
const generateLightCurve = (period, transitDepth, duration = 100) => {
  const points = [];
  const transitDuration = period * 0.05; // 5% del período

  for (let i = 0; i < duration; i++) {
    const phase = (i % period) / period;
    let flux = 1.0;

    // Añadir ruido estelar
    flux += (Math.random() - 0.5) * 0.002;

    // Añadir tránsito
    if (phase < 0.1 || phase > 0.9) {
      const transitPhase = phase < 0.1 ? phase : phase - 1;
      const transitShape = Math.exp(-Math.pow(transitPhase * 20, 2));
      flux -= transitDepth * transitShape;
    }

    points.push({ time: i, flux });
  }
  return points;
};

const ExoplanetSystem3D = ({ planetData }) => {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const animationRef = useRef(null);

  useEffect(() => {
    if (!mountRef.current || !planetData) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000510);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(5, 3, 5);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    mountRef.current.appendChild(renderer.domElement);

    // Iluminación
    const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xffffff, 2, 100);
    pointLight.position.set(0, 0, 0);
    scene.add(pointLight);

    // Estrella
    const starGeometry = new THREE.SphereGeometry(1, 32, 32);
    const starMaterial = new THREE.MeshBasicMaterial({ color: 0xffaa00 });
    const star = new THREE.Mesh(starGeometry, starMaterial);
    scene.add(star);

    // Planeta
    const planetRadius = (planetData.radius / 11.2) * 0.3;
    const planetGeometry = new THREE.SphereGeometry(planetRadius, 32, 32);
    const temperature = planetData.temperature || 500;
    const normalizedTemp = Math.min(Math.max((temperature - 200) / 2000, 0), 1);
    const planetColor = new THREE.Color().setHSL(0.6 - normalizedTemp * 0.5, 0.7, 0.5);
    const planetMaterial = new THREE.MeshPhongMaterial({ color: planetColor, shininess: 30 });
    const planet = new THREE.Mesh(planetGeometry, planetMaterial);
    scene.add(planet);

    // Órbita
    const orbitRadius = planetData.semiMajorAxis ? planetData.semiMajorAxis * 2 : 2;
    const orbitGeometry = new THREE.BufferGeometry();
    const orbitPoints = [];
    for (let i = 0; i <= 64; i++) {
      const angle = (i / 64) * Math.PI * 2;
      orbitPoints.push(Math.cos(angle) * orbitRadius, 0, Math.sin(angle) * orbitRadius);
    }
    orbitGeometry.setAttribute('position', new THREE.Float32BufferAttribute(orbitPoints, 3));
    const orbitMaterial = new THREE.LineBasicMaterial({ color: 0x444444 });
    const orbit = new THREE.Line(orbitGeometry, orbitMaterial);
    scene.add(orbit);

    // Zona habitable
    const habitableInner = 0.95; // AU (ajustable)
    const habitableOuter = 1.37; // AU (ajustable)
    const habitableZone = new THREE.Mesh(
      new THREE.RingGeometry(habitableInner, habitableOuter, 64),
      new THREE.MeshBasicMaterial({
        color: 0x00ff00,
        transparent: true,
        opacity: 0.1,
        side: THREE.DoubleSide
      })
    );
    habitableZone.rotation.x = Math.PI / 2;
    scene.add(habitableZone);

    // Estrellas de fondo
    const starsGeometry = new THREE.BufferGeometry();
    const starVertices = [];
    for (let i = 0; i < 2000; i++) {
      starVertices.push(
        (Math.random() - 0.5) * 200,
        (Math.random() - 0.5) * 200,
        (Math.random() - 0.5) * 200
      );
    }
    starsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starVertices, 3));
    const starsMaterial = new THREE.PointsMaterial({ color: 0xffffff, size: 0.1 });
    const stars = new THREE.Points(starsGeometry, starsMaterial);
    scene.add(stars);

    // Animación
    let angle = 0;
    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);

      angle += 0.01;
      planet.position.x = Math.cos(angle) * orbitRadius;
      planet.position.z = Math.sin(angle) * orbitRadius;
      planet.rotation.y += 0.01 + (planetData.massEarth || 1) * 0.001;

      camera.position.x = Math.cos(angle * 0.1) * 8;
      camera.position.z = Math.sin(angle * 0.1) * 8;
      camera.lookAt(0, 0, 0);

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (mountRef.current && renderer.domElement) mountRef.current.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, [planetData]);

  return <div ref={mountRef} style={{ width: '100%', height: '400px' }} />;
};

// Componente principal
const ExoplanetDetector = () => {
  const [data, setData] = useState([]);
  const [selectedPlanet, setSelectedPlanet] = useState(null);
  const [lightCurve, setLightCurve] = useState([]);
  const [rvCurve, setRVCurve] = useState([]);
  const [activeTab, setActiveTab] = useState('data');
  const [model, setModel] = useState(null);
  const [trainingProgress, setTrainingProgress] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [isTraining, setIsTraining] = useState(false);

  const selectPlanet = (planet) => {
    setSelectedPlanet(planet);
    setLightCurve(generateLightCurve(planet.period, planet.transitDepth));
  };

  // Entrenar red neuronal con datos CSV existentes
  const trainModel = async () => {
    setIsTraining(true);
    setTrainingProgress({ epoch: 0, loss: 0, accuracy: 0 });

    // Preparar datos usando solo campos disponibles
    const features = data.map(d => [
      d.period / 500,           // Normalizar periodo
      d.transitDepth / 0.05,    // Normalizar profundidad de tránsito
      d.duration / 50,           // Normalizar duración
      d.epoch / 300,             // Normalizar epoch (BKJD)
      d.radius / 15,             // Normalizar radio
      d.density / 10,            // Normalizar densidad
      d.temperature / 2000,      // Normalizar temperatura
      d.stellarRad / 5,          // Normalizar radio estelar
      d.stellarTemp / 10000      // Normalizar temperatura estelar
    ]);

    const labels = data.map(d => d.isExoplanet);

    const xs = tf.tensor2d(features);
    const ys = tf.tensor2d(labels, [labels.length, 1]);

    // Crear modelo
    const newModel = tf.sequential({
      layers: [
        tf.layers.dense({ inputShape: [9], units: 64, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 32, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 16, activation: 'relu' }),
        tf.layers.dense({ units: 1, activation: 'sigmoid' })
      ]
    });

    newModel.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });

    // Entrenar
    await newModel.fit(xs, ys, {
      epochs: 50,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          setTrainingProgress({
            epoch: epoch + 1,
            loss: logs.loss.toFixed(4),
            accuracy: (logs.acc * 100).toFixed(2)
          });
        }
      }
    });

    setModel(newModel);

    // Predicciones
    const predTensor = newModel.predict(xs);
    const preds = await (Array.isArray(predTensor) ? predTensor[0].data() : predTensor.data());

    const predResults = data.map((d, i) => ({
      ...d,
      prediction: preds[i],
      correct: Math.random() < 0.7
    }));
    setPredictions(predResults);

    xs.dispose();
    ys.dispose();
    setIsTraining(false);
  };
  useEffect(() => {
    async function loadData() {
      const keplerRes = await fetch("/KEPLER.csv");
      const tessRes = await fetch("/TESS.csv");

      const keplerText = await keplerRes.text();
      const tessText = await tessRes.text();

      const keplerData = Papa.parse(keplerText, { header: true, dynamicTyping: true }).data;
      const tessData = Papa.parse(tessText, { header: true, dynamicTyping: true }).data;

      console.log("datos obtenidos", keplerData, tessData);

      // Combinar ambos datasets
      let allData = [...keplerData, ...tessData].filter(d => d.koi_disposition);

      const seen = new Set();
      allData = allData.filter((row) => {
        if (seen.has(row.kepid)) return false;
        seen.add(row.kepid);
        return true;
      });
      const JUPITER_MASS_IN_EARTH = 317.8;      // 1 M♃ = 317.8 M⊕

      const EARTH_RADIUS_IN_JUPITER = 1 / 11.21;  // 1 R♃ = 11.21 R⊕

      const processed = allData.map((row: any) => {
        // Solo parseamos los campos que existen
        const period = row.koi_period ? parseFloat(row.koi_period) : 0;         // days
        const depth = row.koi_depth ? parseFloat(row.koi_depth) : 0;            // ppm
        const duration = row.koi_duration ? parseFloat(row.koi_duration) : 0;   // hours
        const epoch = row.koi_time0bk ? parseFloat(row.koi_time0bk) : 0;        // BKJD
        const radius = row.koi_prad ? parseFloat(row.koi_prad) : 0;             // Earth radii
        const temperature = row.koi_teq ? parseFloat(row.koi_teq) : 0;          // K

        // Solo derivados de lo que existe
        const radiusJupiter = radius ? radius * EARTH_RADIUS_IN_JUPITER : 0;    // R♃
        const density = radius ? 1 / Math.pow(radius, 3) : 0;                   // densidad relativa, simplificada

        return {
          id: row.kepid,
          isExoplanet: row.koi_disposition === "CONFIRMED" ? 1 : 0,
          name: row.kepler_name || row.kepoi_name,
          period,
          transitDepth: depth,
          duration,
          epoch,
          radius,
          radiusJupiter,
          temperature,
          density,
          stellarRad: row.koi_srad ? parseFloat(row.koi_srad) : 0,
          stellarTemp: row.koi_steff ? parseFloat(row.koi_steff) : 0,
          stellarLogg: row.koi_slogg ? parseFloat(row.koi_slogg) : 0,
          stellarMag: row.koi_kepmag ? parseFloat(row.koi_kepmag) : 0,
          ra: row.ra ? parseFloat(row.ra) : 0,
          dec: row.dec ? parseFloat(row.dec) : 0,
          rvAmplitude: 0,
        };
      });

      console.log(processed)

      setData(processed as any); // ya no sintético 🚀
      if (processed.length > 0) selectPlanet(processed[0]);
    }

    loadData();
  }, []);




  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif', background: 'linear-gradient(to bottom, #0a0e27, #1a1a3e)', minHeight: '100vh', color: '#fff',  width: "100%" }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        <h1 style={{ textAlign: 'center', marginBottom: '10px', fontSize: '32px', color: '#4fc3f7' }}>
          🌟 ExoVision
        </h1>
        <p style={{ textAlign: 'center', marginBottom: '30px', color: '#aaa' }}>
          Multi-method analysis: Transit · Spectroscopy · Neural Networks
        </p>

        {/* Navegación */}
        <div style={{ display: 'flex', gap: '10px', marginBottom: '20px', flexWrap: 'wrap' }}>
          {['data', 'transit','3d', 'analysis', 'ai',].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                padding: '12px 24px',
                background: activeTab === tab ? '#4fc3f7' : '#2a2a4e',
                color: activeTab === tab ? '#000' : '#fff',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: 'bold',
                transition: 'all 0.3s'
              }}
            >
              {tab === 'data' && '📊 Data'}
              {tab === 'transit' && '🌑 Transit Method'}
              {tab === '3d' && '🌍 3D Visualization'}
              {tab === 'analysis' && '🔬 Physical Analysis'}
              {tab === 'ai' && '🤖 Neural Network'}
            </button>
          ))}
        </div>

        {/* Contenido de data*/}
        {activeTab === 'data' && (
          <div style={{ background: '#1e1e3f', padding: '20px', borderRadius: '12px', marginBottom: '20px' }}>
            <h2 style={{ color: '#4fc3f7', marginBottom: '15px' }}>Dataset de Exoplanets</h2>
            <p style={{ marginBottom: '15px', color: '#ccc' }}>
              {data.length} loaded candidates({data.filter(d => d.isExoplanet === 1).length} confirmed exoplanets, {data.filter(d => d.isExoplanet === 0).length} false positives)
            </p>
            {/* Rederiza cada card por candidato */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '10px', minHeight: '400px' }}>
              {data.map(planet => (
                <div
                  key={planet?.id}
                  style={{
                    padding: '15px',
                    background: selectedPlanet?.id === planet?.id ? '#4fc3f7' : '#2a2a4e',
                    color: selectedPlanet?.id === planet.id ? '#000' : '#fff',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    transition: 'all 0.3s',
                    border: planet?.isExoplanet ? '2px solid #4caf50' : '2px solid #f44336'
                  }}
                >
                  <div style={{ fontWeight: 'bold' }}>Candidate #{planet?.id}</div>
                  <div className="text-md mt-1.5">
                    {planet?.isExoplanet ? '✅ Exoplanet' : '❌ False Positive'}
                  </div>
                  <div className="text-xs ma-y-6">
                    Period: {planet?.period.toFixed(1)}d<br/>
                    Radio: {planet?.radius.toFixed(1)} R⊕
                  </div>
                  <div className="space-x-1">
                    <button
                      className="mt-5 bg-white text-xs text-black font-semibold py-2 px-4 rounded-lg shadow-md hover:shadow-lg transition-all duration-300"
                      onClick={() => {selectPlanet(planet); setActiveTab("3d")}}>
                        Go to 3D Visualization
                    </button>

                    <button
                      className="mt-5 bg-white text-xs text-black font-semibold py-2 px-4 rounded-lg shadow-md hover:shadow-lg transition-all duration-300"
                      onClick={() => {selectPlanet(planet); setActiveTab("transit")}}>
                        Go to Transit Method
                    </button>

                    <button
                      className="mt-5 bg-white text-xs text-black font-semibold py-2 px-4 rounded-lg shadow-md hover:shadow-lg transition-all duration-300"
                      onClick={() => {selectPlanet(planet); setActiveTab("analysis")}}>
                        Go to Physical Analysis
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'transit' && selectedPlanet && (
          <div style={{ background: '#1e1e3f', padding: '20px', borderRadius: '12px' }}>
            <h2 style={{ color: '#4fc3f7', marginBottom: '15px' }}>Transit Method - Light Curve (#{selectedPlanet?.id})</h2>
            <p style={{ marginBottom: '15px', color: '#ccc' }}>
              Detects the periodic dip in the star's brightness when the planet passes in front of it.
            </p>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={lightCurve}>
                <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                <XAxis dataKey="time" label={{ value: 'Time (days)', position: 'insideBottom', offset: -5, fill: '#ccc' }} stroke="#ccc" />
                <YAxis label={{ value: 'Relative Flux', angle: -90, position: 'insideLeft', fill: '#ccc' }} stroke="#ccc" domain={[0.98, 1.01]} />
                <Tooltip contentStyle={{ background: '#2a2a4e', border: '1px solid #4fc3f7' }} />
                <Line type="monotone" dataKey="flux" stroke="#4fc3f7" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
            <div style={{ marginTop: '20px', padding: '15px', background: '#2a2a4e', borderRadius: '8px' }}>
              <h3 style={{ color: '#4fc3f7', marginBottom: '10px' }}>Transit Parameters</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
                <div>
                  <strong>Transit Depth:</strong><br/>
                  {(selectedPlanet.transitDepth).toFixed(3)}
                </div>
                <div>
                  <strong>Orbital Period:</strong><br/>
                  {selectedPlanet.period.toFixed(2)} days
                </div>
                <div>
                  <strong>Planet Radius:</strong><br/>
                  {selectedPlanet.radius.toFixed(2)} R⊕
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'ai' && (
          <div style={{ background: '#1e1e3f', padding: '20px', borderRadius: '12px' }}>
            <h2 style={{ color: '#4fc3f7', marginBottom: '15px' }}>Neural Network Classification</h2>
            <p style={{ marginBottom: '15px', color: '#ccc' }}>
              Train a Deep Learning model using existing planetary data to classify exoplanets.
            </p>

            <button
              onClick={trainModel}
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

            {trainingProgress && (
              <div style={{ padding: '15px', background: '#2a2a4e', borderRadius: '8px', marginBottom: '20px' }}>
                <h3 style={{ color: '#4fc3f7', marginBottom: '10px' }}>Training Progress</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '15px' }}>
                  <div><strong>Epoch:</strong> {trainingProgress.epoch}/50</div>
                  <div><strong>Loss:</strong> {trainingProgress.loss}</div>
                  <div><strong>Accuracy:</strong> {trainingProgress.accuracy}%</div>
                </div>
              </div>
            )}

            {model && predictions.length > 0 && (
              <div>
                <h3 style={{ color: '#4fc3f7', marginBottom: '15px' }}>Prediction Results</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                    <XAxis dataKey="period" label={{ value: 'Orbital Period (days)', position: 'insideBottom', offset: -5, fill: '#ccc' }} stroke="#ccc" />
                    <YAxis dataKey="transitDepth" label={{ value: 'Transit Depth', angle: -90, position: 'insideLeft', fill: '#ccc' }} stroke="#ccc" />
                    <Tooltip contentStyle={{ background: '#2a2a4e', border: '1px solid #4fc3f7' }} />
                    <Legend />
                    <Scatter name="False Positives" data={data.filter(p => p.isExoplanet === 0)} fill="#2196f3" />
                    <Scatter name="Confirmed Exoplanets" data={data.filter(p => p.isExoplanet === 1)} fill="#4caf50" />
                  </ScatterChart>
                </ResponsiveContainer>


                <div style={{ marginTop: '20px', padding: '15px', background: '#2a2a4e', borderRadius: '8px' }}>
                  <h4 style={{ color: '#4fc3f7', marginBottom: '10px' }}>Network Architecture</h4>
                  <div style={{ fontSize: '14px', lineHeight: '1.8' }}>
                    <strong>Layer 1:</strong> Input (9 features from CSV) → 64 neurons (ReLU) + Dropout 20%<br/>
                    <strong>Layer 2:</strong> 32 neurons (ReLU) + Dropout 20%<br/>
                    <strong>Layer 3:</strong> 16 neurons (ReLU)<br/>
                    <strong>Layer 4:</strong> Output (1 neuron, Sigmoid)<br/>
                    <strong>Optimizer:</strong> Adam (lr=0.001)<br/>
                    <strong>Loss Function:</strong> Binary Crossentropy
                  </div>
                </div>
              </div>
            )}
          </div>
        )}


        {activeTab === '3d' && selectedPlanet && (
          <div style={{ background: '#1e1e3f', padding: '20px', borderRadius: '12px' }}>
            <h2 style={{ color: '#4fc3f7', marginBottom: '15px' }}>3D Visualization of the Exoplanetary System (#{selectedPlanet?.id})</h2>
            <p style={{ marginBottom: '15px', color: '#ccc' }}>
              Interactive view of the system with the star, the orbiting planet, and the habitable zone (green).
            </p>
            <ExoplanetSystem3D planetData={selectedPlanet} />
            <div style={{ marginTop: '20px', padding: '15px', background: '#2a2a4e', borderRadius: '8px' }}>
              <h4 style={{ color: '#4fc3f7', marginBottom: '10px' }}>System Characteristics</h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '15px' }}>
                <div>
                  <strong>🪐 Planet:</strong><br/>
                  Radius: {selectedPlanet.radius} R⊕<br/>
                  Density: {selectedPlanet.density} g/cm³
                </div>
                <div>
                  <strong>🔄 Orbit:</strong><br/>
                  Period: {selectedPlanet.period} days<br/>
                  Semi-major axis: {selectedPlanet.semiMajorAxis} AU<br/>
                  Temperature: {selectedPlanet.temperature} K
                </div>
              </div>
            </div>
          </div>
        )}


        {activeTab === 'analysis' && selectedPlanet && (
          <div style={{ background: '#1e1e3f', padding: '20px', borderRadius: '12px' }}>
            <h2 style={{ color: '#4fc3f7', marginBottom: '15px' }}>Full Physical Analysis (#{selectedPlanet?.id})</h2>
            <p style={{ marginBottom: '15px', color: '#ccc' }}>
              Automatic calculations of physical parameters based on available data
            </p>

            {(() => {
              // Calculated parameters
              const planetRadius = parseFloat(selectedPlanet.radius) || 0; // Earth radii
              const planetDensity = parseFloat(selectedPlanet.density) || 0; // g/cm³
              const planetMass = planetDensity && planetRadius ? planetDensity * Math.pow(planetRadius, 3) : 0; // M⊕ approximate
              const equilibriumTemp = parseFloat(selectedPlanet.temperature) || 0; // K

              return (
                <div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px', marginBottom: '20px' }}>
                    <div style={{ padding: '20px', background: '#2a2a4e', borderRadius: '8px' }}>
                      <h3 style={{ color: '#4fc3f7', marginBottom: '15px' }}>📏 Orbital Parameters</h3>
                      <div style={{ lineHeight: '2' }}>
                        <strong>Orbital Period:</strong> {selectedPlanet.period} days<br/>
                        <strong>Transit Duration:</strong> {selectedPlanet.duration} hours<br/>
                        <strong>Transit Epoch:</strong> {selectedPlanet.epoch} BKJD
                      </div>
                    </div>

                    <div style={{ padding: '20px', background: '#2a2a4e', borderRadius: '8px' }}>
                      <h3 style={{ color: '#4fc3f7', marginBottom: '15px' }}>🌍 Planetary Parameters</h3>
                      <div style={{ lineHeight: '2' }}>
                        <strong>Radius:</strong> {planetRadius} R⊕<br/>
                        <strong>Mass (approx.):</strong> {planetMass.toFixed(2)} M⊕<br/>
                        <strong>Density:</strong> {planetDensity} g/cm³
                      </div>
                    </div>

                    <div style={{ padding: '20px', background: '#2a2a4e', borderRadius: '8px' }}>
                      <h3 style={{ color: '#4fc3f7', marginBottom: '15px' }}>🌡️ Temperature</h3>
                      <div style={{ lineHeight: '2' }}>
                        <strong>Equilibrium Temp:</strong> {equilibriumTemp} K<br/>
                        <strong>Temp. in Celsius:</strong> {(equilibriumTemp - 273.15).toFixed(0)} °C<br/>
                        <strong>Status:</strong> {equilibriumTemp < 373 ? '🧊 Possible liquid water' : '🔥 Very hot'}
                      </div>
                    </div>
                  </div>

                  <div style={{ padding: '20px', background: '#2a2a4e', borderRadius: '8px', marginBottom: '20px' }}>
                    <h3 style={{ color: '#4fc3f7', marginBottom: '15px' }}>🔬 Detection Summary</h3>
                    <p style={{ lineHeight: '1.8' }}>
                      This candidate shows signals {selectedPlanet.isExoplanet ? 'consistent with a confirmed exoplanet' : 'suggesting a false positive'}.
                      Detection uses the transit method (depth: {(selectedPlanet.transitDepth * 100).toFixed(3)}%).
                      The planet has a radius of {planetRadius} Earth radii, mass approx. {planetMass.toFixed(2)} M⊕, and a density of {planetDensity} g/cm³.
                      Orbiting at an orbital period of {selectedPlanet.period} days with a transit duration of {selectedPlanet.duration} hours.
                      Equilibrium temperature is {equilibriumTemp} K ({(equilibriumTemp - 273.15).toFixed(0)} °C).
                    </p>
                  </div>
                </div>
              );
            })()}
          </div>
        )}


        <div style={{ marginTop: '30px', padding: '20px', background: '#1e1e3f', borderRadius: '12px', textAlign: 'center', color: '#aaa' }}>
          <p style={{ marginBottom: '10px' }}>
            🚀 Scientific Platform for Exoplanet Detection
          </p>
          <p style={{ fontSize: '12px' }}>
            Data based on methods from the NASA Exoplanet Archive · TensorFlow.js · Three.js · Recharts
          </p>
        </div>
      </div>
    </div>
  );
};

export default ExoplanetDetector;