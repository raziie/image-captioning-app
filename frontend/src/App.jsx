import { useState, useEffect, useRef } from 'react'
import toast, { Toaster } from 'react-hot-toast';
import axios from 'axios'
import './App.css'

function App() {
  const [image, setImage] = useState(null);
  const [response, setResponse] = useState({});
  const [error, setError] = useState("");
  const [isGenerated, setIsGenerated] = useState(false);
  const audioRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [copied, setCopied] = useState(false);
  const [loading, setLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    fetch("http://127.0.0.1:5000/ping")
      .then(res => res.json())
      .then(data => console.log("Connected to backend:", data))
      .catch(err => console.error("Backend not responding:", err));
  }, []);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const updateTime = () => setCurrentTime(audio.currentTime);
    const updateDuration = () => setDuration(audio.duration);

    audio.addEventListener("timeupdate", updateTime);
    audio.addEventListener("loadedmetadata", updateDuration);

    return () => {
      audio.removeEventListener("timeupdate", updateTime);
      audio.removeEventListener("loadedmetadata", updateDuration);
    };
  }, [response.audio]);

  const handleUpload = async (e) => {
    e.preventDefault();

    if (!image) {
      setError("Please select an image first.");
      return;
    }
    const formData = new FormData();
    formData.append("image", image);

    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:5000/predict", formData);
      setResponse(res.data);
      setIsGenerated(true);
    } catch (err) {
      console.error(err);
      setError("Caption generation failed.");
      setResponse({});
      setIsGenerated(false);
    } finally {
      setLoading(false);
    }
  };

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (audio.paused) {
      audio.play();
      setIsPlaying(true);
    } else {
      audio.pause();
      setIsPlaying(false);
    }
  };

  const toggleMute = () => {
    const audio = audioRef.current;
    if (!audio) return;

    audio.muted = !audio.muted;
    setIsMuted(audio.muted);
  };

  const handleSeek = (e) => {
    const audio = audioRef.current;
    if (!audio) return;

    const newTime = (e.target.value / 100) * duration;
    audio.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const formatTime = (time) => {
    const min = Math.floor(time / 60);
    const sec = Math.floor(time % 60)
      .toString()
      .padStart(2, "0");
    return `${min}:${sec}`;
  };

  const handleCopy = () => {
    if (response.caption) {
      navigator.clipboard.writeText(response.caption)
        .then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 2000); // Reset after 2 sec
        })
        .catch(() => {
          alert("Copy failed. Try again.");
        });
    }
  };

  return (
    <div id="webcrumbs">
      <div className="w-full min-w-[320px] bg-white text-gray-800 overflow-hidden p-6 md:p-10 max-w-4xl mx-auto">
        <div className="flex flex-col space-y-8">
          {/* Header */}
          <div className="text-center">
            <h1 className="text-3xl md:text-4xl font-bold mb-2">Image Caption Generator</h1>
            <p className="text-gray-600">Upload an image to generate a caption with audio narration</p>
          </div>

          {/* Upload Form */}
          <div className="bg-gray-50 p-6 rounded-lg shadow-sm border border-gray-200">
            <form onSubmit={handleUpload} className="space-y-4">
              <div className="space-y-2">
                <div className="flex flex-col items-center justify-center w-full gap-4">
                  <label
                    onDragOver={(e) => {
                      e.preventDefault();
                      setIsDragging(true);
                    }}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={(e) => {
                      e.preventDefault();
                      setIsDragging(false);
                      const file = e.dataTransfer.files[0];
                      if (file && file.type.startsWith("image/")) {
                        setImage(file);
                        setError("");
                      } else {
                        setError("Please upload a valid image file (PNG, JPG, JPEG)");
                      }
                    }}
                    className={`flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-lg cursor-pointer transition-colors duration-300 ${isDragging ? "border-blue-500 bg-blue-50" : "border-gray-300 bg-white hover:bg-gray-50"
                      }`}>
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                      <span className="material-symbols-outlined text-gray-500 text-3xl mb-2">
                        upload_file
                      </span>
                      <p className="mb-2 text-sm text-gray-500">
                        <span className="font-semibold">Click to upload </span>
                        or drag and drop
                      </p>
                      <p className="text-xs text-gray-500">PNG, JPG or JPEG</p>
                    </div>
                    <input
                      id="image-upload"
                      type="file"
                      className="hidden"
                      accept="image/png, image/jpeg, image/jpg"
                      onChange={(e) => {
                        setImage(e.target.files[0]);
                        setError("");
                      }}
                    />
                  </label>
                  {error && (
                    <p className="text-red-600 text-sm mt-2 text-center">{error}</p>
                  )}
                  <div className="flex flex-col gap-4">
                    {image && <img
                      className="object-cover w-full h-full"
                      src={URL.createObjectURL(image)}
                      alt="uploaded image"
                    />}
                  </div>
                </div>
              </div>
              <button
                type="submit"
                disabled={loading}
                className={`w-full px-4 py-3 rounded-lg transition-all duration-300 ease-in-out flex items-center justify-center shadow-md focus:outline-none focus:ring-2 focus:ring-offset-2 bg-sky-600
                                ${loading
                    ? "text-white cursor-not-allowed"
                    : "text-white hover:-translate-y-1 hover:bg-sky-700 focus:ring-sky-500"}
                              `}
              >
                <span className="material-symbols-outlined mr-2">
                  {loading ? "hourglass_top" : "auto_awesome"}
                </span>
                {loading ? "Generating..." : "Generate Caption"}
              </button>

            </form>
          </div>

          {/* Preview Section */}
          {isGenerated && <div className="space-y-6">
            {/* Caption */}
            <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
              <h3 className="text-lg font-medium mb-2">Generated Caption</h3>
              {/* Additional Actions */}
              {isGenerated && <div className="flex flex-wrap gap-3 justify-between">
                  <p className="text-gray-700 mb-4">{response.caption}</p>
                  <button className="flex gap-3 items-center" onClick={handleCopy}>
                      {copied && (
                          <span className="text-xs text-gray-500 animate-fade-in z-50">
                            Copied to clipboard!
                          </span>
                        )}
                      <span className="material-symbols-outlined mr-2 text-gray-600">content_copy</span>
                  </button>
              </div>}
              {/* Audio Player */}
              {isGenerated && (
                <div className="bg-gray-50 p-4 rounded-md shadow-sm space-y-4">
                  <audio
                    ref={audioRef}
                    src={`http://localhost:5000/${response.audio}`}
                    preload="metadata"
                    className="hidden"
                    onEnded={() => setIsPlaying(false)}
                  />
                  <div className="flex items-center justify-between gap-3">
                    <button
                      onClick={togglePlay}
                      className="flex p-2 rounded-full bg-sky-600 text-white hover:bg-sky-700 transition"
                    >
                      <span className="material-symbols-outlined">{isPlaying ? "pause" : "play_arrow"}</span>
                    </button>

                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={duration ? (currentTime / duration) * 100 : 0}
                      onChange={handleSeek}
                      className="w-full h-2 bg-gray-200 rounded-lg cursor-pointer"
                    />

                    <span className="text-sm text-gray-500 whitespace-nowrap">
                      {formatTime(currentTime)}
                    </span>
                    <span className="text-sm text-gray-500 whitespace-nowrap">
                      /
                    </span>
                    <span className="text-sm text-gray-500 whitespace-nowrap">
                      {formatTime(duration)}
                    </span>

                    <button
                      onClick={toggleMute}
                      className="flex p-2 rounded-full hover:bg-gray-200 transition"
                    >
                      <span className="material-symbols-outlined">
                        {isMuted ? "volume_off" : "volume_up"}
                      </span>
                    </button>

                    <a
                      href={`http://localhost:5000/${response.audio}`}
                      download="caption-audio.mp3"
                      className="flex p-2 rounded-full hover:bg-gray-200 transition"
                    >
                      <span className="material-symbols-outlined">download</span>
                    </a>
                  </div>
                </div>
              )}
            </div>
            <div className="flex justify-center mt-6">
              <img
                src={`http://localhost:5000/${response.attention_map_plot}`}
                alt="Attention Map"
                className="max-w-full max-h-[400px] rounded-lg shadow-lg border border-gray-300"
              />
            </div>
          </div>}

          {/* Footer */}
          <div className="text-center text-gray-500 text-sm pt-4 border-t border-gray-200">
            <p>
              Powered by Image Captioning App •{" "}
              <a href="https://github.com/raziie/image-captioning-app" className="text-sky-600 hover:underline">
                Learn more
              </a>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
