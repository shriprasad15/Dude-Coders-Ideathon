import { useState, useCallback, useRef } from 'react'
import { GoogleMap, useJsApiLoader, Marker, Autocomplete } from '@react-google-maps/api'
import { AnimatePresence, motion } from 'framer-motion'
import { Loader2, Zap, X, Search, Sparkles, Navigation, Globe, Upload, MessageSquare } from 'lucide-react'
import axios from 'axios'
import clsx from 'clsx'

const GOOGLE_MAPS_API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY || "";

const mapContainerStyle = {
    width: '100vw',
    height: '100vh',
};

const defaultCenter = {
    lat: 21.5030457,
    lng: 70.45946829
};

const mapOptions = {
    mapTypeId: 'hybrid',
    disableDefaultUI: true,
    zoomControl: false,
    tilt: 0,
};

const libraries = ['places'];

function App() {
    const { isLoaded, loadError } = useJsApiLoader({
        id: 'google-map-script',
        googleMapsApiKey: GOOGLE_MAPS_API_KEY,
        libraries
    })

    // State
    const [map, setMap] = useState(null)
    const [searchResult, setSearchResult] = useState(null)
    const [selectedLocation, setSelectedLocation] = useState(null)
    const [isConfirming, setIsConfirming] = useState(false)
    const [isAnalyzing, setIsAnalyzing] = useState(false)
    const [result, setResult] = useState(null)

    // Smart Search Context
    const [showSmartSearch, setShowSmartSearch] = useState(false);
    const [file, setFile] = useState(null);
    const [smartQuery, setSmartQuery] = useState("");
    const [isSmartSearching, setIsSmartSearching] = useState(false);
    const [chatHistory, setChatHistory] = useState([
        { type: 'bot', text: "Hello! Upload a file or just ask me to find a location." }
    ]);
    const chatEndRef = useRef(null);

    // Map Handlers
    const onLoad = useCallback(function callback(map) {
        setMap(map)
    }, [])

    const onUnmount = useCallback(function callback(map) {
        setMap(null)
    }, [])

    // Search
    const onSearchLoad = (autocomplete) => {
        setSearchResult(autocomplete);
    }

    const onPlaceChanged = () => {
        if (searchResult !== null) {
            const place = searchResult.getPlace();
            if (place.geometry && place.geometry.location) {
                const location = {
                    lat: place.geometry.location.lat(),
                    lng: place.geometry.location.lng()
                };

                map.panTo(location);
                map.setZoom(19);
                setSelectedLocation(location);
                setIsConfirming(true);
                setResult(null);
            }
        }
    }

    const onMapClick = useCallback((e) => {
        const lat = e.latLng.lat();
        const lng = e.latLng.lng();
        setSelectedLocation({ lat, lng });
        setIsConfirming(true);
        setResult(null);
    }, [])

    const handleConfirm = async () => {
        setIsConfirming(false);
        setIsAnalyzing(true);

        try {
            const response = await axios.post('/api/analyze', {
                lat: selectedLocation.lat,
                lon: selectedLocation.lng
            });

            setResult(response.data);
        } catch (error) {
            console.error("Analysis failed", error);
            alert("Analysis failed. Check console. Is backend running?");
        } finally {
            setIsAnalyzing(false);
        }
    };

    const closePopup = () => {
        setSelectedLocation(null);
        setResult(null);
        setIsConfirming(false);
    };

    const handleZoomIn = () => map?.setZoom(map.getZoom() + 1);
    const handleZoomOut = () => map?.setZoom(map.getZoom() - 1);

    const handleFileChange = (e) => {
        if (e.target.files) {
            setFile(e.target.files[0]);
        }
    }

    const handleSmartSearchSubmit = async () => {
        if (!smartQuery.trim() && !file) return;

        // Add User Message
        const displayText = smartQuery.trim() || `Analyzing file: ${file.name}...`;
        const userMsg = { type: 'user', text: displayText };
        setChatHistory(prev => [...prev, userMsg]);
        setSmartQuery("");
        setIsSmartSearching(true);

        // Add Temporary Loading Message
        setChatHistory(prev => [...prev, { type: 'bot', text: "Thinking...", isLoading: true }]);

        try {
            const formData = new FormData();
            formData.append('query', smartQuery.trim()); // Send empty string if empty
            if (file) {
                formData.append('file', file);
            }

            const res = await axios.post('/api/smart-search', formData);

            // Remove loading message
            setChatHistory(prev => prev.filter(msg => !msg.isLoading));

            if (res.data.lat && res.data.lon) {
                const loc = { lat: res.data.lat, lng: res.data.lon };
                map.panTo(loc);
                map.setZoom(19);
                setSelectedLocation(loc);
                setIsConfirming(true);
                setResult(null);

                // Add Bot Success Message
                setChatHistory(prev => [...prev, {
                    type: 'bot',
                    text: `Found it! Moving to ${res.data.location_name || "target location"}.`
                }]);
            } else {
                setChatHistory(prev => [...prev, { type: 'bot', text: "I couldn't find coordinates in that data." }]);
            }
        } catch (e) {
            console.error(e);
            setChatHistory(prev => prev.filter(msg => !msg.isLoading));

            let errorMessage = "Something went wrong. Please try again.";

            // Check for Rate Limit (429) or Specific Details
            if (e.response && (e.response.status === 429 || (e.response.data && e.response.data.detail && e.response.data.detail.includes("429")))) {
                errorMessage = "Rate limit exceeded (Free Tier). Please wait ~10 seconds and try again.";
            } else if (e.response && e.response.data && e.response.data.detail) {
                // Show clean error if available
                errorMessage = typeof e.response.data.detail === 'string'
                    ? `Error: ${e.response.data.detail.substring(0, 100)}...`
                    : "An error occurred with the AI service.";
            }

            setChatHistory(prev => [...prev, { type: 'bot', text: errorMessage, isError: true }]);
        } finally {
            setIsSmartSearching(false);
        }
    }

    // Auto-scroll to bottom of chat
    const scrollToBottom = () => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }

    // Watch for chat updates
    useState(() => {
        scrollToBottom();
    }, [chatHistory]);

    if (loadError) return <div className="text-red-500 text-center mt-20">Map Load Error: {loadError.message}</div>
    if (!isLoaded) return <div className="flex items-center justify-center h-screen bg-black text-white">Loading Solar Experience...</div>

    return (
        <div className="relative w-screen h-screen overflow-hidden bg-black font-sans selection:bg-blue-500/30">

            {/* 1. Fullscreen Map (Z-0) */}
            <div className="absolute inset-0 z-0">
                <GoogleMap
                    mapContainerStyle={mapContainerStyle}
                    center={defaultCenter}
                    zoom={18}
                    onClick={onMapClick}
                    options={mapOptions}
                    onLoad={onLoad}
                    onUnmount={onUnmount}
                >
                    {selectedLocation && <Marker position={selectedLocation} />}
                </GoogleMap>
            </div>

            {/* 2. Floating Sidebar (Glass Panel) (Z-20) */}
            <motion.div
                initial={{ x: -100, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                className="absolute left-6 top-6 bottom-6 w-80 z-20 flex flex-col pointer-events-none"
            >
                {/* Glass Container */}
                <div className="flex-1 rounded-3xl bg-black/40 backdrop-blur-xl border border-white/10 shadow-2xl overflow-hidden flex flex-col pointer-events-auto ring-1 ring-white/10">

                    {/* Header */}
                    <div className="p-6 border-b border-white/5">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center shadow-lg shadow-blue-600/20">
                                <Zap className="text-white w-6 h-6 fill-current" />
                            </div>
                            <div>
                                <h1 className="text-lg font-bold tracking-tight text-white leading-none">Solar Potential</h1>
                                <span className="text-xs text-blue-300 font-medium">Analysis Tool</span>
                            </div>
                        </div>
                    </div>

                    {/* Search Section */}
                    <div className="p-6 space-y-4">
                        <div className="relative group">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 group-focus-within:text-blue-400 transition-colors" />
                            <Autocomplete onLoad={onSearchLoad} onPlaceChanged={onPlaceChanged}>
                                <input
                                    type="text"
                                    placeholder="Search location..."
                                    className="w-full bg-white/5 border border-white/10 rounded-xl py-3 pl-10 pr-4 text-sm text-white placeholder-gray-500 
                                     focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:bg-white/10 transition-all shadow-inner"
                                />
                            </Autocomplete>
                        </div>

                        {/* Smart Search Toggle */}
                        <button
                            onClick={() => setShowSmartSearch(!showSmartSearch)}
                            className="w-full flex items-center justify-between px-4 py-3 rounded-xl bg-gradient-to-r from-blue-600/20 to-purple-600/20 border border-blue-500/30 hover:border-blue-500/50 transition-all group"
                        >
                            <div className="flex items-center gap-2">
                                <Sparkles className="w-4 h-4 text-blue-400 group-hover:text-blue-300" />
                                <span className="text-sm font-medium text-blue-100">Smart Search</span>
                            </div>
                            <span className="text-[10px] bg-blue-500/20 px-2 py-0.5 rounded text-blue-300">New</span>
                        </button>

                        {/* Smart Search Panel (Expandable) */}
                        <AnimatePresence>
                            {showSmartSearch && (
                                <motion.div
                                    initial={{ height: 0, opacity: 0 }}
                                    animate={{ height: "auto", opacity: 1 }}
                                    exit={{ height: 0, opacity: 0 }}
                                    className="overflow-hidden"
                                >
                                    <div className="bg-white/5 rounded-xl border border-white/5 flex flex-col max-h-[400px]">

                                        {/* Top: File Upload */}
                                        <div className="p-3 border-b border-white/5 bg-white/5">
                                            <label className={clsx("w-full py-2 rounded-lg border border-white/5 flex items-center justify-center gap-2 text-xs font-medium transition-colors cursor-pointer",
                                                file ? "bg-green-500/20 text-green-400 border-green-500/30" : "bg-white/5 hover:bg-white/10 text-gray-300"
                                            )}>
                                                <Upload className="w-3 h-3" />
                                                {file ? file.name : "Upload Data (Optional)"}
                                                <input type="file" onChange={handleFileChange} accept=".csv,.xlsx,.xls" className="hidden" />
                                            </label>
                                        </div>

                                        {/* Middle: Chat Log */}
                                        <div className="flex-1 overflow-y-auto p-3 space-y-3 min-h-[200px] custom-scrollbar">
                                            {chatHistory.map((msg, i) => (
                                                <div key={i} className={clsx("flex flex-col max-w-[85%]", msg.type === 'user' ? "ml-auto items-end" : "mr-auto items-start")}>
                                                    <div className={clsx("px-3 py-2 rounded-2xl text-xs leading-relaxed",
                                                        msg.type === 'user'
                                                            ? "bg-blue-600 text-white rounded-br-none"
                                                            : "bg-white/10 text-gray-200 rounded-bl-none"
                                                    )}>
                                                        {msg.text}
                                                    </div>
                                                </div>
                                            ))}
                                            <div ref={chatEndRef} />
                                        </div>

                                        {/* Bottom: Input */}
                                        <div className="p-3 border-t border-white/5 bg-white/5">
                                            <div className="relative">
                                                <input
                                                    type="text"
                                                    value={smartQuery}
                                                    onChange={(e) => setSmartQuery(e.target.value)}
                                                    onKeyDown={(e) => e.key === 'Enter' && handleSmartSearchSubmit()}
                                                    placeholder="Ask anything..."
                                                    disabled={isSmartSearching}
                                                    className="w-full bg-black/50 border border-white/10 rounded-full py-2.5 pl-4 pr-10 text-xs text-white focus:outline-none focus:border-blue-500/50 disabled:opacity-50"
                                                />
                                                <button
                                                    onClick={handleSmartSearchSubmit}
                                                    disabled={isSmartSearching || (!smartQuery.trim() && !file)}
                                                    className="absolute right-1 top-1/2 -translate-y-1/2 w-7 h-7 rounded-full bg-blue-600 hover:bg-blue-500 flex items-center justify-center text-white disabled:opacity-50 disabled:bg-gray-700 transition-colors"
                                                >
                                                    {isSmartSearching ? <Loader2 className="w-3 h-3 animate-spin" /> : <Navigation className="w-3 h-3" />}
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>

                    {/* Status Cards */}
                    <div className="flex-1 overflow-y-auto px-6 pb-6 space-y-3 custom-scrollbar">
                        <div className="p-4 rounded-2xl bg-white/5 border border-white/5 hover:bg-white/10 transition-colors cursor-default backdrop-blur-sm">
                            <div className="flex items-center gap-3 mb-2">
                                <div className="p-1.5 rounded-lg bg-green-500/20 text-green-400">
                                    <Globe className="w-3 h-3" />
                                </div>
                                <span className="text-xs font-medium uppercase tracking-wider text-gray-400">System Ready</span>
                            </div>
                            <div className="flex items-end justify-between">
                                <span className="text-2xl font-bold text-white">Online</span>
                                <div className="flex items-center gap-1.5">
                                    <span className="relative flex h-2 w-2">
                                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                                        <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="p-4 border-t border-white/5 bg-black/20 text-center">
                        <p className="text-[10px] text-gray-500 font-medium">Powered by Gemini & YOLOv8</p>
                    </div>
                </div>
            </motion.div>

            {/* 3. Floating Zoom Controls */}
            <div className="absolute right-6 bottom-8 z-20 flex flex-col gap-2">
                <button onClick={handleZoomIn} className="w-10 h-10 rounded-xl bg-black/40 backdrop-blur-xl border border-white/10 flex items-center justify-center text-white hover:bg-white/10 transition-colors">
                    <span className="text-xl">+</span>
                </button>
                <button onClick={handleZoomOut} className="w-10 h-10 rounded-xl bg-black/40 backdrop-blur-xl border border-white/10 flex items-center justify-center text-white hover:bg-white/10 transition-colors">
                    <span className="text-xl">-</span>
                </button>
            </div>

            {/* 4. Overlays (Dialogs & Results) */}
            <AnimatePresence>
                {isConfirming && selectedLocation && (
                    <motion.div
                        initial={{ y: 50, opacity: 0, scale: 0.9 }}
                        animate={{ y: 0, opacity: 1, scale: 1 }}
                        exit={{ y: 20, opacity: 0 }}
                        className="absolute bottom-10 left-1/2 -translate-x-1/2 z-30"
                    >
                        <div className="px-6 py-4 rounded-2xl bg-black/60 backdrop-blur-xl border border-white/10 shadow-2xl flex items-center gap-6 ring-1 ring-white/20">
                            <div>
                                <h3 className="text-sm font-bold text-white">Run Analysis?</h3>
                                <div className="flex items-center gap-1 text-xs text-gray-300">
                                    <Navigation className="w-3 h-3" />
                                    {selectedLocation.lat.toFixed(4)}, {selectedLocation.lng.toFixed(4)}
                                </div>
                            </div>
                            <div className="flex gap-2">
                                <button onClick={() => setIsConfirming(false)} className="px-4 py-2 rounded-xl text-xs font-semibold bg-white/10 hover:bg-white/20 transition-colors text-white">Cancel</button>
                                <button onClick={handleConfirm} className="px-4 py-2 rounded-xl text-xs font-semibold bg-blue-600 hover:bg-blue-500 shadow-lg shadow-blue-600/30 transition-all text-white">Scan Area</button>
                            </div>
                        </div>
                    </motion.div>
                )}

                {isAnalyzing && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="absolute inset-0 z-40 bg-black/40 backdrop-blur-md flex flex-col items-center justify-center p-8 text-center"
                    >
                        <div className="w-20 h-20 rounded-3xl bg-white/5 border border-white/10 flex items-center justify-center mb-6 relative overflow-hidden">
                            <div className="absolute inset-0 bg-gradient-to-tr from-blue-500 to-purple-500 opacity-20 animate-spin" />
                            <Loader2 className="w-8 h-8 text-white animate-spin relative z-10" />
                        </div>
                        <h2 className="text-2xl font-bold text-white mb-2">Analyzing Terrain</h2>
                        <p className="text-blue-200">Processing satellite imagery...</p>
                    </motion.div>
                )}

                {result && (
                    <motion.div
                        initial={{ opacity: 0, y: 100, scale: 0.9 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 100, scale: 0.9 }}
                        transition={{ type: "spring", damping: 25, stiffness: 300 }}
                        className="absolute inset-4 md:inset-20 z-50 flex items-center justify-center pointer-events-none"
                    >
                        <div className="bg-black/80 backdrop-blur-2xl border border-white/10 shadow-2xl rounded-3xl overflow-hidden w-full max-w-4xl max-h-full flex flex-col md:flex-row pointer-events-auto ring-1 ring-white/20">
                            {/* Image Side */}
                            <div className="w-full md:w-[55%] bg-black relative min-h-[300px] md:h-full group">
                                {result.image_base64 ? (
                                    <img src={`data:image/jpeg;base64,${result.image_base64}`} className="w-full h-full object-cover" />
                                ) : (
                                    <div className="flex items-center justify-center h-full text-gray-500">No Image Available</div>
                                )}

                                {/* Overlay Gradient */}
                                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-60"></div>

                                <div className="absolute bottom-6 left-6 right-6">
                                    <span className={clsx("inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider mb-2",
                                        result.has_solar ? "bg-green-500/80 text-white shadow-lg shadow-green-500/40" : "bg-red-500/80 text-white shadow-lg shadow-red-500/40"
                                    )}>
                                        {result.has_solar ? <Zap className="w-3 h-3 fill-current" /> : <X className="w-3 h-3" />}
                                        {result.has_solar ? "Solar Detected" : "No Solar Found"}
                                    </span>
                                    <h3 className="text-white font-bold text-lg drop-shadow-md">{result.metadata?.source || "Satellite Capture"}</h3>
                                    <p className="text-gray-300 text-xs">{result.metadata?.capture_date}</p>
                                </div>

                                <button onClick={closePopup} className="absolute top-4 left-4 p-2 bg-black/40 hover:bg-black/60 rounded-full text-white backdrop-blur-md transition-colors">
                                    <X className="w-5 h-5" />
                                </button>
                            </div>

                            {/* Details Side */}
                            <div className="w-full md:w-[45%] p-8 flex flex-col bg-gradient-to-b from-white/5 to-transparent">
                                <h2 className="text-2xl font-bold text-white mb-6">Inference Report</h2>

                                <div className="space-y-6">
                                    <div className="bg-white/5 rounded-2xl p-4 border border-white/5">
                                        <label className="text-xs text-gray-400 uppercase tracking-wider font-semibold">Confidence</label>
                                        <div className="flex items-baseline gap-2 mt-1">
                                            <span className={clsx("text-3xl font-bold", result.confidence > 0.7 ? "text-green-400" : "text-yellow-400")}>
                                                {(result.confidence * 100).toFixed(1)}%
                                            </span>
                                            <span className="text-sm text-gray-500">probability</span>
                                        </div>
                                        <div className="w-full bg-white/10 h-2 rounded-full mt-3 overflow-hidden">
                                            <motion.div
                                                initial={{ width: 0 }}
                                                animate={{ width: `${result.confidence * 100}%` }}
                                                transition={{ duration: 1, delay: 0.2 }}
                                                className={clsx("h-full rounded-full", result.confidence > 0.7 ? "bg-green-500" : "bg-yellow-500")}
                                            />
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="bg-white/5 rounded-2xl p-4 border border-white/5">
                                            <label className="text-xs text-gray-400 uppercase tracking-wider font-semibold">Lat</label>
                                            <p className="text-white font-mono mt-1">{selectedLocation?.lat.toFixed(5)}</p>
                                        </div>
                                        <div className="bg-white/5 rounded-2xl p-4 border border-white/5">
                                            <label className="text-xs text-gray-400 uppercase tracking-wider font-semibold">Lng</label>
                                            <p className="text-white font-mono mt-1">{selectedLocation?.lng.toFixed(5)}</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="mt-auto pt-8">
                                    <button onClick={closePopup} className="w-full py-4 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-bold text-sm shadow-xl shadow-blue-600/20 transition-all hover:scale-[1.02] active:scale-[0.98]">
                                        Done
                                    </button>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}

export default App
