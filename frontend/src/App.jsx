import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

function App() {
    const [image, setImage] = useState(null);
    const [response, setResponse] = useState({});
    const [error, setError] = useState("");
    const [isGenerated, setIsGenerated] = useState(false);

    useEffect(() => {
      fetch("http://127.0.0.1:5000/ping")
        .then(res => res.json())
        .then(data => console.log("✅ Connected to backend:", data))
        .catch(err => console.error("❌ Backend not responding:", err));
    }, []);

    const handleUpload = async (e) => {
        e.preventDefault();

        if (!image) {
            setError("Please select an image first.");
            return;
        }
        const formData = new FormData();
        formData.append("image", image);

        try {
          const res = await axios.post("http://127.0.0.1:5000/predict", formData);
          setResponse(res.data);
          setIsGenerated(true);
        } catch (err) {
          console.error(err);
          setError("Caption generation failed.");
          setResponse({});
          setIsGenerated(false);
        }
    };

    return (
    <>
    <div>
      <h1>Image Captioning</h1>
      <form onSubmit={handleUpload}>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
              setImage(e.target.files[0]);
              setError("");
          }}
        />
        <button type="submit">Generate</button>
      </form>
      {error && <p className="error-text">{error}</p>}
      {image && <img className="image" src={URL.createObjectURL(image)} alt="uploaded image" />}
      {isGenerated && <p>{response.caption}</p>}
      {isGenerated && <audio key={response.audio} controls autoPlay>
        <source src={`http://localhost:5000/${response.audio}`} type="audio/mpeg" />
        Your browser does not support the audio element.
      </audio>}
    </div>
    </>
    )
}

export default App
