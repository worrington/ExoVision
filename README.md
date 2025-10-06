# ExoVision: Exoplanet Predictor 🌌

A web application to predict whether an observed planet could be an **exoplanet** using a **TensorFlow.js** machine learning model. It allows training the model with historical data and predicting new planets based on their observable characteristics.

---

## 🛠 Requirements

Before running the project, make sure you have installed:

- [Node.js](https://nodejs.org/) >= 18
- Modern browser compatible with TensorFlow.js (Chrome, Firefox, Edge)
- Git (to clone the repository)
- Visual Studio Code or your preferred code editor

---

## 📥 Installation

1. Clone the repository:

```bash
git clone https://github.com/worrington/ExoVision.git
```

2. Enter the project directory:

```bash
cd ExoVision
```

3. Install the dependencies:

```bash
npm install
```

---

## ⚡️ Usage

1. Start the development server:

```bash
npm run dev
```

2. Open your browser at:

```
http://localhost:3000
```

3. Train the model from the interface (if there is no saved model) and then use the form to predict new exoplanets.

---

## 📦 Project Structure

- `src/lib/exoplanetModel.js`: Functions to train, load, and predict using the TensorFlow.js model
- `src/components/ExoplanetForm.jsx`: Form to enter planet data and get predictions
- `src/data/exoplanets.json`: Sample dataset with confirmed and unconfirmed planets (Data from NASA Exoplanet Archive)
- `public/`: Static files
- `pages/`: Next.js routes

---

## 🔧 Libraries Used

- [Next.js](https://nextjs.org/) – React framework for SSR/SSG
- [React](https://reactjs.org/) – UI library
- [TensorFlow.js](https://www.tensorflow.org/js) – Machine Learning in JavaScript
- [Tailwind CSS](https://tailwindcss.com/) – Styling framework

---

## 💾 Model Storage

The trained model is automatically saved in **localStorage**:

```
tensorflowjs_models/exoplanet-model
```

If it exists, the form will be automatically displayed to predict new planets.

---

## 📄 License

This project is licensed under the MIT License.
Feel free to use, modify, and share it.
