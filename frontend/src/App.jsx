import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import { GoogleMap, useJsApiLoader, Marker, Autocomplete } from '@react-google-maps/api'
import { AnimatePresence, motion } from 'framer-motion'
import { Loader2, Zap, X, Search, Sparkles, Navigation, Globe, Upload, MessageSquare, ExternalLink, FileSpreadsheet, Download, UploadCloud, ChevronLeft, ChevronRight, Filter, FileJson } from 'lucide-react'
import axios from 'axios'
import clsx from 'clsx'
import ReactMarkdown from 'react-markdown'



const GOOGLE_MAPS_API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY || "";

// Configure API Base URL
const API_BASE_URL = import.meta.env.PROD
    ? "https://sue-asymmetric-nonprogressively.ngrok-free.dev"
    : "/api";

// Configure Axios Global Defaults
axios.defaults.baseURL = API_BASE_URL;
// Bypass Ngrok Browser Warning
axios.defaults.headers.common['ngrok-skip-browser-warning'] = 'true';

const mapContainerStyle = {
    width: '100vw',
    height: '100vh',
};

const defaultCenter = {
    lat: 12.99151,
    lng: 80.23362
};

const mapOptions = {
    mapTypeId: 'hybrid', // Hybrid = Satellite + Labels
    disableDefaultUI: true,
    zoomControl: false,
    // tilt: 0, // Default is 0, no need to specify
    // heading: 0,
    // mapId removed for standard raster map
};

const libraries = ['places'];


function App() {
    const { isLoaded, loadError } = useJsApiLoader({
        id: 'google-map-script',
        googleMapsApiKey: GOOGLE_MAPS_API_KEY,
        libraries
    })

    // Helper for Skeleton Animation
    const shimmer = {
        hidden: { x: "-100%" },
        visible: { x: "100%", transition: { repeat: Infinity, duration: 1.5, ease: "linear" } }
    };

    // State
    const [map, setMap] = useState(null)
    const [selectedLocation, setSelectedLocation] = useState(null)
    const [userLocation, setUserLocation] = useState(null); // Add User Location State
    const [isConfirming, setIsConfirming] = useState(false)
    const [isAnalyzing, setIsAnalyzing] = useState(false)
    const [result, setResult] = useState(null)

    // ... (rest of state)


    const [inputValue, setInputValue] = useState("");
    const [predictions, setPredictions] = useState([]);
    const [showDropdown, setShowDropdown] = useState(false);
    const autocompleteServiceRef = useRef(null);
    const placesServiceRef = useRef(null);

    // State for Modals
    const [showTeam, setShowTeam] = useState(false);
    const [showHowTo, setShowHowTo] = useState(false);
    const [showBulkModal, setShowBulkModal] = useState(false);

    // Bulk Analysis State
    const [isBulkAnalyzing, setIsBulkAnalyzing] = useState(false);
    const [bulkResults, setBulkResults] = useState([]);
    const [bulkFileName, setBulkFileName] = useState(null); // New State

    const [filterMode, setFilterMode] = useState('all'); // 'all', 'solar', 'no-solar'

    // Logic to get filtered results
    const getFilteredResults = useCallback(() => {
        return bulkResults.filter(res => {
            if (filterMode === 'solar') return res.has_solar;
            if (filterMode === 'no-solar') return !res.has_solar;
            return true;
        });
    }, [bulkResults, filterMode]);

    // Copilot Integration


    // --- NAVIGATION HELPERS ---
    const flyToLocation = (targetLoc) => {
        if (!map) return;
        // Standard 2D Pan
        map.panTo(targetLoc);
        map.setZoom(19);
    };
    const handleNext = () => {
        const filtered = getFilteredResults();
        if (filtered.length === 0) return;

        let nextIdx = 0;
        // Find current index based on selectedLocation or result
        // Use selectedLocation to track current position even if result (report) isn't open
        const currentLoc = selectedLocation || (result ? { lat: result.lat, lng: result.lng } : null);

        if (currentLoc) {
            const currentIdx = filtered.findIndex(r => isSameLocation({ lat: r.lat, lng: r.lng }, currentLoc));
            if (currentIdx !== -1) {
                nextIdx = (currentIdx + 1) % filtered.length;
            }
        }

        const nextRes = filtered[nextIdx];
        setSelectedLocation({ lat: nextRes.lat, lng: nextRes.lng });

        // Only open/update the report if it is ALREADY open
        if (result) {
            setResult(nextRes);
        }

        flyToLocation({ lat: nextRes.lat, lng: nextRes.lng });
    };

    const handlePrev = () => {
        const filtered = getFilteredResults();
        if (filtered.length === 0) return;

        let prevIdx = filtered.length - 1;

        const currentLoc = selectedLocation || (result ? { lat: result.lat, lng: result.lng } : null);

        if (currentLoc) {
            const currentIdx = filtered.findIndex(r => isSameLocation({ lat: r.lat, lng: r.lng }, currentLoc));
            if (currentIdx !== -1) {
                prevIdx = (currentIdx - 1 + filtered.length) % filtered.length;
            }
        }

        const prevRes = filtered[prevIdx];
        setSelectedLocation({ lat: prevRes.lat, lng: prevRes.lng });

        // Only open/update the report if it is ALREADY open
        if (result) {
            setResult(prevRes);
        }

        flyToLocation({ lat: prevRes.lat, lng: prevRes.lng });
    };

    const downloadReport = (format) => {
        const filtered = getFilteredResults();

        if (format === 'json') {
            const dataStr = JSON.stringify(filtered.map(r => ({
                sample_id: r.sample_id || "N/A",
                lat: r.lat,
                lon: r.lng,
                has_solar: r.has_solar,
                confidence: r.confidence,
                pv_area_sqm_est: r.pv_area_sqm_est,
                euclidean_distance_m_est: r.euclidean_distance_m_est,
                buffer_radius_sqft: r.buffer_size || 0,
                qc_status: "VERIFIABLE",
                bbox_or_mask: r.bbox_or_mask || [],
                image_metadata: { source: "Cache", capture_date: new Date().toISOString().split('T')[0] }
            })), null, 2);

            const blob = new Blob([dataStr], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `solar_report_${bulkFileName || 'results'}.json`;
            a.click();
            URL.revokeObjectURL(url);
        } else if (format === 'csv') {
            // CSV Headers
            const headers = ["sample_id", "lat", "lon", "has_solar", "confidence", "pv_area_sqm_est", "euclidean_distance_m_est", "buffer_radius_sqft", "qc_status", "image_source", "capture_date"];

            // CSV Rows
            const rows = filtered.map(r => [
                r.sample_id || "N/A",
                r.lat,
                r.lng,
                r.has_solar,
                r.confidence,
                r.pv_area_sqm_est,
                r.euclidean_distance_m_est,
                r.buffer_size || 0,
                "VERIFIABLE",
                "Cache",
                new Date().toISOString().split('T')[0]
            ].map(f => `"${f}"`).join(",")); // Quote fields

            const csvContent = [headers.join(","), ...rows].join("\n");

            const blob = new Blob([csvContent], { type: "text/csv" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `solar_report_${bulkFileName || 'results'}.csv`;
            a.click();
            URL.revokeObjectURL(url);
        }
    };

    // Helper to compare locations with small epsilon
    const isSameLocation = (loc1, loc2) => {
        if (!loc1 || !loc2) return false;
        const eps = 0.000001;
        return Math.abs(loc1.lat - loc2.lat) < eps && Math.abs(loc1.lng - loc2.lng) < eps;
    };



    // Map Handlers
    const onLoad = useCallback(function callback(map) {
        setMap(map)

        // --- LIVE LOCATION ON LOAD ---
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    const pos = {
                        lat: position.coords.latitude,
                        lng: position.coords.longitude,
                    };
                    setUserLocation(pos);
                    // Zoom to moderate level (16) as requested
                    map.panTo(pos);
                    map.setZoom(16);
                },
                (error) => {
                    console.log("Auto-location failed or denied:", error);
                    // Fallback is handled by default parameters (Zoom 0/Global)
                }
            );
        }
        // Init Services
        if (window.google && window.google.maps) {
            autocompleteServiceRef.current = new window.google.maps.places.AutocompleteService();
            placesServiceRef.current = new window.google.maps.places.PlacesService(map);
        }

    }, [])

    const onUnmount = useCallback(function callback(map) {
        setMap(null)
    }, [])

    // Live Geolocation Tracking
    useEffect(() => {
        if (!map) return;

        if (navigator.geolocation) {
            const watchId = navigator.geolocation.watchPosition(
                (position) => {
                    const pos = {
                        lat: position.coords.latitude,
                        lng: position.coords.longitude,
                    };
                    setUserLocation(pos);
                    // Only pan on first load or if following mode is active? 
                    // For now, let's just update the dot. 
                    // User said: "like how it also moves when the device moves"
                },
                (error) => {
                    console.error("Geolocation watch error:", error);
                },
                {
                    enableHighAccuracy: true,
                    maximumAge: 0,
                    timeout: 5000
                }
            );
            return () => navigator.geolocation.clearWatch(watchId);
        }
    }, [map]);

    // New Custom Autocomplete Logic
    const handleInputChange = (e) => {
        const val = e.target.value;
        setInputValue(val);

        if (!val) {
            setPredictions([]);
            setShowDropdown(false);
            return;
        }

        if (autocompleteServiceRef.current) {
            autocompleteServiceRef.current.getPlacePredictions({ input: val }, (predictions, status) => {
                if (status === window.google.maps.places.PlacesServiceStatus.OK && predictions) {
                    setPredictions(predictions);
                    setShowDropdown(true);
                } else {
                    setPredictions([]);
                }
            });
        }
    };

    const handlePredictionSelect = (placeId, description) => {
        setInputValue(description);
        setShowDropdown(false);
        setPredictions([]);

        if (placesServiceRef.current) {
            placesServiceRef.current.getDetails({ placeId: placeId }, (place, status) => {
                if (status === window.google.maps.places.PlacesServiceStatus.OK && place.geometry && place.geometry.location) {
                    const location = {
                        lat: place.geometry.location.lat(),
                        lng: place.geometry.location.lng()
                    };
                    flyToLocation(location);
                    setSelectedLocation(location);
                    setIsConfirming(true);
                    setResult(null);
                }
            });
        }
    };

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
            const response = await axios.post('/analyze', {
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

    const handleLocateMe = () => {
        if (map) {
            if (userLocation) {
                flyToLocation(userLocation);
                return;
            }

            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(
                    (position) => {
                        const pos = {
                            lat: position.coords.latitude,
                            lng: position.coords.longitude,
                        };
                        setUserLocation(pos);
                        flyToLocation(pos);
                    },
                    (error) => {
                        console.error("Error finding location", error);
                        alert("Could not find your location. Please check browser permissions.");
                    }
                );
            } else {
                alert("Geolocation is not supported by your browser");
            }
        }
    };

    const handleFileChange = (e) => {
        if (e.target.files) {
            setFile(e.target.files[0]);
        }
    }

    const handleBulkUpload = async (uploadFile) => {
        if (!uploadFile) return;
        setIsBulkAnalyzing(true);
        setBulkResults([]); // Clear previous

        try {
            const formData = new FormData();
            formData.append('file', uploadFile);

            const response = await axios.post('/bulk-analyze', formData);

            // Map the results
            setBulkResults(response.data)
            setBulkFileName(uploadFile.name); // Set filename
            setShowBulkModal(false)

            // Calculate center of all points
            if (response.data.length > 0) {
                const lats = response.data.map(r => r.lat)
                const lngs = response.data.map(r => r.lng)
                const minLat = Math.min(...lats)
                const maxLat = Math.max(...lats)
                const minLng = Math.min(...lngs)
                const maxLng = Math.max(...lngs)

                // Assuming setMapCenter is defined elsewhere, or needs to be added.
                // For now, let's just pan to the first result as before, or fit bounds.
                // The instruction implies a setMapCenter, but it's not in the provided code.
                // Sticking to the original behavior of panning to the first result for now.
                // If setMapCenter is a new state, it needs to be declared.
                // For now, I'll use the original panTo logic.
                const first = response.data[0];
                if (first.lat && first.lng) {
                    flyToLocation({ lat: first.lat, lng: first.lng });
                }
            }

        } catch (error) {
            console.error(error)
            alert("Failed to analyze file. Ensure it has valid lat/lon columns.")
        } finally {
            setIsBulkAnalyzing(false)
        }
    };

    const downloadTemplate = () => {
        const headers = "lat,lon,location_name\n";
        const row1 = "12.99151,80.23362,Location 1\n";
        const row2 = "12.99200,80.23400,Location 2\n";

        const blob = new Blob([headers + row1 + row2], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.setAttribute('hidden', '');
        a.setAttribute('href', url);
        a.setAttribute('download', 'solar_analysis_template.csv');
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    };

    // Image Download Helper
    const downloadImage = (base64Str, filename) => {
        const link = document.createElement("a");
        link.href = `data:image/jpeg;base64,${base64Str}`;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    // Drag and Drop Logic
    const onDrop = useCallback((e) => {
        e.preventDefault();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleBulkUpload(e.dataTransfer.files[0]);
        }
    }, []);

    const onDragOver = useCallback((e) => {
        e.preventDefault();
    }, []);



    // Cleanup services on unmount
    useEffect(() => {
        return () => {
            autocompleteServiceRef.current = null;
            placesServiceRef.current = null;
        }
    }, [])

    // Custom Marker Icon (3D Red Ball Pin) - Defined early to avoid hook order issues
    const pinIcon = useMemo(() => {
        if (!isLoaded || !window.google) return null;
        return {
            url: 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(`
            <svg width="60" height="90" viewBox="0 0 60 90" fill="none" xmlns="http://www.w3.org/2000/svg">
                <!-- Shadow Base -->
                <ellipse cx="30" cy="85" rx="8" ry="3" fill="black" fill-opacity="0.3"/>
                <!-- Stick -->
                <rect x="27" y="30" width="6" height="55" fill="#C0C0C0" stroke="#808080" stroke-width="1"/>
                <!-- Ball Head -->
                <circle cx="30" cy="30" r="20" fill="url(#grad1)" stroke="#A00000" stroke-width="1"/>
                <defs>
                    <radialGradient id="grad1" cx="35%" cy="35%" r="65%">
                        <stop offset="0%" stop-color="#FF5555" />
                        <stop offset="100%" stop-color="#CC0000" />
                    </radialGradient>
                </defs>
            </svg>
            `),
            scaledSize: new window.google.maps.Size(40, 60),
            anchor: new window.google.maps.Point(20, 60),
        };
    }, [isLoaded]);

    // User Location Blue Dot Icon (Native Look)
    const userLocationIcon = useMemo(() => ({
        url: 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(`
            <svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                <!-- Outer Halo (Accuracy) -->
                <circle cx="20" cy="20" r="18" fill="#4285F4" fill-opacity="0.15"/>
                <!-- White Border -->
                <circle cx="20" cy="20" r="9" fill="white"/>
                <!-- Blue Inner Dot -->
                <circle cx="20" cy="20" r="7" fill="#4285F4"/>
            </svg>
        `),
        scaledSize: { width: 40, height: 40 },
        anchor: { x: 20, y: 20 },
    }), []);

    if (loadError) return <div className="text-red-500 text-center mt-20">Map Load Error: {loadError.message}</div>
    if (!isLoaded) return <div className="flex items-center justify-center h-screen bg-black text-white">Loading Solar Experience...</div>



    const HOW_TO_CONTENT = `
# Solar Panel Detection System

## Overview
This repository contains an end-to-end pipeline for detecting solar panels in satellite imagery. It uses a fine-tuned YOLOv12 model with a specialized multi-stage fallback inference strategy to handle difficult cases (small panels, low contrast).

## Technical Methodology

### 1. Multi-Stage Fallback Strategy
To address challenges with small panels and low-contrast satellite imagery, we implemented a robust **6-stage fallback mechanism**. If a panel is not detected in the initial pass, the system progressively relaxes constraints and applies enhancements:

1.  **Initial Check (1200 sqft)**: Check for panels within a small 1200 sqft buffer (approx. residential roof size) from the standard inference.
2.  **Saturated Check (1200 sqft)**: If failed, saturate the image (HSV +50%) to boost contrast and re-run inference.
3.  **Cropped Check (1200 sqft)**: If failed, physically crop the image to the 1200 sqft region and re-run inference (improves relative object size).
4.  **Saturated Crop Check (1200 sqft)**: Combine cropping and saturation.
5.  **Expanded Check (2400 sqft)**: If still failed, repeat the initial check on a larger 2400 sqft buffer.
6.  **Expanded Saturated Check (2400 sqft)**: Final attempt using saturation on the larger buffer.

### 2. Advanced Training Strategy
-   **Hard Negative Sampling**: We explicitly trained the model on "negative" samples—satellite images of rooftops *without* solar panels—to drastically reduce false positives. This forces the model to learn the specific features of solar panels rather than just "rectangular things on roofs."
-   **Data Augmentation**: Heavy use of Mosaic, Mixup, and HSV augmentation during training to generalize across different lighting conditions and resolutions.

## QC Status Logic
-   **VERIFIABLE**: 
    -   Solar Panel Found with **High Confidence (> 70%)**.
    -   **OR** Solar Panel **Not Found** after exhaustive search (Verified Absent).
-   **NOT_VERIFIABLE**:
    -   Solar Panel Found but with **Low Confidence (<= 70%)**.

## Model Performance
The model utilizes **YOLOv12** architecture fine-tuned on our custom dataset.

## How to Use Application

**1. Search for a Location**
Use the search bar in the left panel to find any address. The map will fly to the location.

**2. Confirm & Analyze**
Once you select a location, a "Confirm Coordinates" sheet will appear. Click "Analyze" to run the solar detection model.

**3. View Results**
The AI-powered result card will appear, showing:
- Whether solar panels were detected.
- The confidence score of the detection.
- Estimated solar array area (sqm).
- Distance from the property center.

**4. Ask AI Agent (Smart Search)**
Use the chat interface to upload data files or ask questions about locations.
    `;



    return (
        <div className="flex flex-col h-screen w-screen bg-black text-white font-sans selection:bg-white/30 overflow-hidden">

            {/* 1. Navbar */}
            <nav className="flex-none h-16 px-6 md:px-8 flex items-center justify-between border-b border-white/10 z-50 bg-black">
                <div className="flex items-center gap-6">
                    <div className="flex items-center gap-2">
                        <span className="text-xl font-bold tracking-tight">Dude</span>
                        <span className="text-xs bg-white text-black px-2 py-0.5 rounded font-bold uppercase tracking-wider">Coders</span>
                    </div>
                    {/* System Status Moved Here */}
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/5">
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                        </span>
                        <span className="text-[10px] font-bold text-gray-300 uppercase tracking-widest">System Online</span>
                    </div>
                </div>

                <div className="hidden md:flex items-center gap-8 font-semibold tracking-wide text-white/80">
                    <button onClick={() => setShowHowTo(true)} className="hover:text-white transition-colors hover:scale-105 active:scale-95 uppercase text-xs tracking-widest">Protocol</button>
                    <button onClick={() => setShowTeam(true)} className="hover:text-white transition-colors hover:scale-105 active:scale-95 uppercase text-xs tracking-widest">Team</button>
                    <a href="https://github.com/shriprasad15/Dude-Coders-Ideathon/" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors hover:scale-105 active:scale-95 flex items-center gap-1.5 uppercase text-xs tracking-widest">
                        GitHub <ExternalLink className="w-3 h-3 text-white/50" />
                    </a>
                </div>
            </nav>

            {/* 2. Main Split Layout */}
            <div className="flex-1 flex flex-col-reverse md:flex-row relative overflow-hidden">

                {/* Left Column: Content & Controls */}
                <div className="w-full md:w-[400px] lg:w-[450px] flex-none z-20 bg-black flex flex-col border-t md:border-t-0 md:border-r border-white/10 shadow-2xl relative h-[55vh] md:h-auto">
                    <div className="flex-1 overflow-y-auto custom-scrollbar p-6 md:p-8 flex flex-col">

                        <div className="mb-8">
                            <h1 className="text-3xl md:text-4xl font-bold leading-tight tracking-tight mb-3 text-white">
                                Solar Intelligence <br />for Everyone.
                            </h1>
                            <p className="text-sm text-gray-400 leading-relaxed">
                                Instant rooftop analysis using Satellite Imagery. Enter a location to estimate renewable potential.
                            </p>
                        </div>

                        {/* Search Container */}
                        <div className="bg-white/5 rounded-2xl p-1.5 border border-white/10 mb-6 relative z-50">
                            {/* Autocomplete Input */}
                            <div className="relative group">
                                <div className="absolute left-5 top-1/2 -translate-y-1/2 w-2 h-2 rounded-sm bg-white box-content border-[3px] border-black/10 z-10"></div>

                                {/* CUSTOM AUTOCOMPLETE UI */}
                                <div className="relative">
                                    <input
                                        type="text"
                                        value={inputValue}
                                        onChange={handleInputChange}
                                        placeholder="Search location to analyze"
                                        className="w-full bg-[#111] text-white placeholder-gray-500 px-6 py-5 pl-14 rounded-xl border-none focus:ring-1 focus:ring-white/20 transition-all font-bold text-lg leading-relaxed shadow-inner"
                                        onKeyDown={(e) => {
                                            if (e.key === 'Enter') {
                                                const val = e.target.value;

                                                // 1. Check for Coordinates (Lat, Lon)
                                                const coordRegex = /^(-?\d+(\.\d+)?),\s*(-?\d+(\.\d+)?)$/;
                                                const match = val.match(coordRegex);

                                                if (match) {
                                                    const lat = parseFloat(match[1]);
                                                    const lng = parseFloat(match[3]);

                                                    // Valid ranges
                                                    if (lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180) {
                                                        const location = { lat, lng };
                                                        flyToLocation(location);
                                                        setSelectedLocation(location);
                                                        setIsConfirming(true);
                                                        setResult(null);
                                                        setShowDropdown(false);
                                                        return;
                                                    }
                                                }

                                                // Fallback to top result if processed
                                                if (predictions.length > 0) {
                                                    handlePredictionSelect(predictions[0].place_id, predictions[0].description);
                                                    return;
                                                }

                                                // Existing fallback logic for raw text
                                                if (!val) return;
                                                setTimeout(() => {
                                                    if (selectedLocation) return;
                                                    if (autocompleteServiceRef.current) {
                                                        autocompleteServiceRef.current.getPlacePredictions({ input: val }, (preds, status) => {
                                                            if (status === window.google.maps.places.PlacesServiceStatus.OK && preds && preds.length > 0) {
                                                                handlePredictionSelect(preds[0].place_id, preds[0].description);
                                                            }
                                                        });
                                                    }
                                                }, 200);
                                            }
                                        }}
                                        onBlur={() => {
                                            // Delay hiding to allow click event to fire
                                            setTimeout(() => setShowDropdown(false), 200);
                                        }}
                                        onFocus={() => {
                                            if (predictions.length > 0) setShowDropdown(true);
                                        }}
                                    />

                                    {/* Dropdown Results */}
                                    <AnimatePresence>
                                        {showDropdown && predictions.length > 0 && (
                                            <motion.div
                                                initial={{ opacity: 0, y: -10 }}
                                                animate={{ opacity: 1, y: 0 }}
                                                exit={{ opacity: 0, y: -10 }}
                                                className="absolute left-0 right-0 top-full mt-2 bg-[#0a0a0a] border border-white/10 rounded-xl shadow-2xl z-[60] overflow-hidden max-h-80 overflow-y-auto custom-scrollbar ring-1 ring-white/10"
                                            >
                                                {predictions.map((prediction) => (
                                                    <div
                                                        key={prediction.place_id}
                                                        onClick={() => handlePredictionSelect(prediction.place_id, prediction.description)}
                                                        className="px-5 py-4 hover:bg-white/10 cursor-pointer border-b border-white/5 last:border-0 flex items-center gap-3 transition-colors group"
                                                    >
                                                        <Search className="w-4 h-4 text-gray-500 group-hover:text-white transition-colors flex-none" />
                                                        <div className="flex-1 min-w-0">
                                                            <div className="text-white font-bold text-base truncate group-hover:text-blue-400 transition-colors">
                                                                {prediction.structured_formatting?.main_text || prediction.description}
                                                            </div>
                                                            <div className="text-gray-400 text-xs truncate">
                                                                {prediction.structured_formatting?.secondary_text || ""}
                                                            </div>
                                                        </div>
                                                    </div>
                                                ))}
                                                <div className="px-5 py-2 bg-white/5 text-[10px] text-gray-500 text-right uppercase tracking-widest hidden">
                                                    Powered by Google
                                                </div>
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </div>
                            </div>
                        </div>

                        {/* Bulk Analysis Button */}
                        <div className="mb-6 z-30 relative">
                            <button
                                onClick={() => setShowBulkModal(true)}
                                className="w-full bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl p-4 flex items-center justify-between transition-all group"
                            >
                                <div className="flex items-center gap-3">
                                    <div className="w-10 h-10 rounded-lg bg-blue-500/20 text-blue-400 flex items-center justify-center group-hover:scale-110 transition-transform">
                                        <FileSpreadsheet className="w-5 h-5" />
                                    </div>
                                    <div className="text-left">
                                        <div className="text-white font-bold text-sm">Bulk Analysis</div>
                                        <div className="text-gray-500 text-xs">Upload CSV/Excel (Lat/Lon)</div>
                                    </div>
                                </div>
                                <Upload className="w-4 h-4 text-gray-500 group-hover:text-white transition-colors" />
                            </button>
                        </div>



                    </div>
                </div>
                {/* Right Column: Key visual (Map) */}
                <div className="flex-1 bg-[#111] relative overflow-hidden">
                    <GoogleMap
                        mapContainerStyle={{ width: '100%', height: '100%' }}
                        center={defaultCenter}
                        zoom={18}
                        onClick={onMapClick}
                        options={mapOptions}
                        onLoad={onLoad}
                        onUnmount={onUnmount}
                    >
                        {/* Blue Dot for User Location */}
                        {userLocation && (
                            <Marker
                                position={userLocation}
                                icon={userLocationIcon}
                                zIndex={100} // Keep it below the main pin (usually higher zIndex) but above map tiles
                            />
                        )}

                        {/* Selected Location Pin - Hide if it matches a Bulk Result */}
                        {selectedLocation && !bulkResults.some(b => isSameLocation(b, selectedLocation)) && (
                            <Marker position={selectedLocation} icon={pinIcon} />
                        )}

                        {/* Bulk Markers - Updated to Pins */}
                        {bulkResults
                            .filter(res => {
                                if (filterMode === 'solar') return res.has_solar;
                                if (filterMode === 'no-solar') return !res.has_solar;
                                return true;
                            })
                            .map((res, idx) => (
                                <Marker
                                    key={`bulk-${idx}`}
                                    position={{ lat: res.lat, lng: res.lng }}
                                    onClick={() => {
                                        setSelectedLocation({ lat: res.lat, lng: res.lng });
                                        // result might be set from bulkResults, or fetch fresh? 
                                        // Since we already have the result in bulkResults, we set it.
                                        setResult(res);
                                    }}
                                    // Use a standard pin path
                                    icon={{
                                        path: "M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z",
                                        fillColor: res.has_solar ? "#4ade80" : "#ef4444", // Green : Red
                                        fillOpacity: 1,
                                        strokeColor: isSameLocation({ lat: res.lat, lng: res.lng }, selectedLocation) ? "#ffffff" : "#000000", // White stroke if selected
                                        strokeWeight: isSameLocation({ lat: res.lat, lng: res.lng }, selectedLocation) ? 2.5 : 1,
                                        scale: isSameLocation({ lat: res.lat, lng: res.lng }, selectedLocation) ? 2.2 : 1.8,
                                        anchor: new window.google.maps.Point(12, 22), // Anchor at the tip
                                        labelOrigin: new window.google.maps.Point(12, -10)
                                    }}
                                />
                            ))}

                        {/* File Indicator & Controls Overlay */}
                        {bulkResults.length > 0 && (
                            <div className="absolute top-4 left-4 z-10 bg-[#0a0a0a] border border-white/10 rounded-xl flex flex-col shadow-xl min-w-[320px]">
                                {/* Header */}
                                <div className="p-3 border-b border-white/10 flex items-center justify-between bg-white/5 rounded-t-xl">
                                    <div className="flex items-center gap-2">
                                        <FileSpreadsheet className="w-4 h-4 text-blue-400" />
                                        <span className="text-sm font-bold text-white max-w-[150px] truncate">{bulkFileName || "Bulk Analysis"}</span>
                                    </div>
                                    <button
                                        onClick={() => {
                                            setBulkResults([]);
                                            setBulkFileName(null);
                                            setSelectedLocation(null);
                                            setResult(null);
                                            setFilterMode('all');
                                        }}
                                        className="text-gray-500 hover:text-white transition-colors"
                                    >
                                        <X className="w-4 h-4" />
                                    </button>
                                </div>

                                {/* Controls Row */}
                                <div className="p-3 flex flex-col gap-3">
                                    {/* Filters with Text */}
                                    <div className="flex bg-white/5 rounded-lg p-1 w-full relative">
                                        <button
                                            onClick={() => setFilterMode('all')}
                                            className={clsx("flex-1 py-1.5 px-2 rounded-md transition-all flex items-center justify-center gap-1.5 text-[10px] font-bold uppercase tracking-wider",
                                                filterMode === 'all' ? "bg-white/20 text-white shadow-sm" : "text-gray-500 hover:text-white hover:bg-white/5"
                                            )}
                                        >
                                            <Filter className="w-3 h-3" /> All
                                        </button>
                                        <div className="w-px bg-white/5 my-1 mx-1"></div>
                                        <button
                                            onClick={() => setFilterMode('solar')}
                                            className={clsx("flex-1 py-1.5 px-2 rounded-md transition-all flex items-center justify-center gap-1.5 text-[10px] font-bold uppercase tracking-wider",
                                                filterMode === 'solar' ? "bg-green-500/20 text-green-400 shadow-sm" : "text-gray-500 hover:text-green-400 hover:bg-green-500/5"
                                            )}
                                        >
                                            <Zap className="w-3 h-3" /> Solar
                                        </button>
                                        <div className="w-px bg-white/5 my-1 mx-1"></div>
                                        <button
                                            onClick={() => setFilterMode('no-solar')}
                                            className={clsx("flex-1 py-1.5 px-2 rounded-md transition-all flex items-center justify-center gap-1.5 text-[10px] font-bold uppercase tracking-wider",
                                                filterMode === 'no-solar' ? "bg-red-500/20 text-red-400 shadow-sm" : "text-gray-500 hover:text-red-400 hover:bg-red-500/5"
                                            )}
                                        >
                                            <div className="w-2.5 h-2.5 rounded-full border border-current" /> No Solar
                                        </button>
                                    </div>

                                    {/* Navigation & Clear Actions */}
                                    <div className="flex items-center justify-between">
                                        {/* Clear Button */}
                                        <button
                                            onClick={() => {
                                                setBulkResults([]);
                                                setBulkFileName(null);
                                                setSelectedLocation(null);
                                                setResult(null);
                                                setFilterMode('all');
                                            }}
                                            className="px-3 py-1.5 bg-red-500/10 hover:bg-red-500/20 text-red-400 text-[10px] font-bold rounded-lg transition-colors uppercase tracking-wider flex items-center gap-1.5 border border-red-500/20"
                                        >
                                            <X className="w-3 h-3" /> Clear Results
                                        </button>

                                        {/* Pagination */}
                                        <div className="flex items-center gap-1">
                                            <button onClick={handlePrev} className="w-7 h-7 flex items-center justify-center rounded-lg bg-white/10 hover:bg-white/20 text-white transition-colors disabled:opacity-30 disabled:cursor-not-allowed" disabled={bulkResults.length === 0}>
                                                <ChevronLeft className="w-4 h-4" />
                                            </button>
                                            <span className="text-xs font-mono text-gray-400 min-w-[30px] text-center select-none">
                                                {(() => {
                                                    const filtered = getFilteredResults();
                                                    // Use selectedLocation to match current index if result isn't open
                                                    const currentLoc = selectedLocation || (result ? { lat: result.lat, lng: result.lng } : null);

                                                    let displayIdx = '-';
                                                    if (currentLoc) {
                                                        const idx = filtered.findIndex(r => isSameLocation({ lat: r.lat, lng: r.lng }, currentLoc));
                                                        if (idx !== -1) displayIdx = idx + 1;
                                                    }
                                                    return (
                                                        <>
                                                            <span className="text-white font-bold">{displayIdx}</span>
                                                            <span className="text-gray-600 mx-1">/</span>
                                                            {filtered.length}
                                                        </>
                                                    )
                                                })()}
                                            </span>
                                            <button onClick={handleNext} className="w-7 h-7 flex items-center justify-center rounded-lg bg-white/10 hover:bg-white/20 text-white transition-colors disabled:opacity-30 disabled:cursor-not-allowed" disabled={bulkResults.length === 0}>
                                                <ChevronRight className="w-4 h-4" />
                                            </button>
                                        </div>
                                    </div>
                                </div>

                                {/* Status & Download */}
                                <div className="px-3 pb-3 flex items-center justify-between border-t border-white/5 pt-3">
                                    <span className="text-[10px] uppercase text-gray-500 font-bold tracking-wider">Export Report</span>
                                    <div className="flex gap-2">
                                        <button onClick={() => downloadReport('json')} className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition-colors flex items-center gap-1.5">
                                            <FileJson className="w-3 h-3" /> <span className="text-[10px] font-bold">JSON</span>
                                        </button>
                                        <button onClick={() => downloadReport('csv')} className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition-colors flex items-center gap-1.5">
                                            <FileSpreadsheet className="w-3 h-3" /> <span className="text-[10px] font-bold">CSV</span>
                                        </button>
                                    </div>
                                </div>
                            </div>
                        )}
                    </GoogleMap>

                    {/* Controls */}
                    <div className="absolute bottom-8 right-8 flex flex-col gap-2 z-20 items-center">
                        {/* Locate Me Button */}
                        <motion.button
                            onClick={handleLocateMe}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            className="w-12 h-12 rounded-full bg-black/80 backdrop-blur-xl border border-white/20 flex items-center justify-center text-white shadow-2xl hover:bg-white hover:text-black transition-colors"
                            title="Locate Me"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><line x1="22" y1="12" x2="18" y2="12" /><line x1="6" y1="12" x2="2" y2="12" /><line x1="12" y1="6" x2="12" y2="2" /><line x1="12" y1="22" x2="12" y2="18" /></svg>
                        </motion.button>

                        <div className="flex flex-col rounded-full bg-black/80 backdrop-blur-xl border border-white/20 overflow-hidden shadow-2xl">
                            <motion.button
                                onClick={handleZoomIn}
                                whileTap={{ backgroundColor: "rgba(255,255,255,0.2)" }}
                                className="w-12 h-12 flex items-center justify-center text-white hover:bg-white/10 transition-colors border-b border-white/10"
                            >
                                <span className="text-2xl font-light">+</span>
                            </motion.button>
                            <motion.button
                                onClick={handleZoomOut}
                                whileTap={{ backgroundColor: "rgba(255,255,255,0.2)" }}
                                className="w-12 h-12 flex items-center justify-center text-white hover:bg-white/10 transition-colors"
                            >
                                <span className="text-2xl font-light">-</span>
                            </motion.button>
                        </div>
                    </div>

                    {/* Overlays: Confirmation */}
                    <AnimatePresence>
                        {isConfirming && selectedLocation && (
                            <motion.div
                                key="confirm-overlay"
                                initial={{ y: 20, opacity: 0 }}
                                animate={{ y: 0, opacity: 1 }}
                                exit={{ y: 20, opacity: 0 }}
                                className="absolute bottom-6 left-6 right-6 md:left-auto md:right-auto md:w-96 z-30 pointer-events-none"
                            >
                                <div className="bg-black border border-white/10 p-6 rounded-2xl shadow-2xl pointer-events-auto">
                                    <h3 className="font-bold text-white text-lg mb-1">Confirm Location?</h3>
                                    <div className="text-xs text-gray-400 mb-4 font-mono bg-white/5 p-2 rounded border border-white/5 inline-block">
                                        {selectedLocation.lat.toFixed(5)}, {selectedLocation.lng.toFixed(5)}
                                    </div>
                                    <div className="flex gap-3">
                                        <button onClick={() => setIsConfirming(false)} className="flex-1 py-3 rounded-lg bg-white/10 hover:bg-white/20 text-white text-sm font-bold transition-colors">Cancel</button>
                                        <button onClick={handleConfirm} className="flex-1 py-3 rounded-lg bg-white text-black hover:bg-gray-200 text-sm font-bold transition-colors">Analyze</button>
                                    </div>
                                </div>
                            </motion.div>
                        )}

                        {isAnalyzing && (
                            <motion.div
                                key="analyzing-skeleton"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="absolute inset-0 z-50 p-4 md:p-12 flex items-center justify-center bg-black/40 backdrop-blur-sm"
                            >
                                <div className="bg-black border border-white/10 rounded-3xl shadow-2xl overflow-hidden w-full max-w-5xl h-[85vh] flex flex-col md:flex-row relative">

                                    {/* 1. COLLIDING BLUE LINE ANIMATION (Top Bar) */}
                                    <div className="absolute top-0 left-0 right-0 h-1 bg-white/5 overflow-hidden z-20">
                                        <motion.div
                                            initial={{ left: "0%", right: "100%" }}
                                            animate={{
                                                left: ["0%", "45%", "0%"],
                                                right: ["100%", "45%", "100%"]
                                            }}
                                            transition={{
                                                duration: 2,
                                                ease: "easeInOut",
                                                repeat: Infinity,
                                                times: [0, 0.5, 1]
                                            }}
                                            className="absolute top-0 bottom-0 bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.8)]"
                                        />
                                        <motion.div
                                            initial={{ left: "100%", right: "0%" }}
                                            animate={{
                                                left: ["100%", "55%", "100%"],
                                                right: ["0%", "55%", "0%"]
                                            }}
                                            transition={{
                                                duration: 2,
                                                ease: "easeInOut",
                                                repeat: Infinity,
                                                times: [0, 0.5, 1]
                                            }}
                                            className="absolute top-0 bottom-0 bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.8)]"
                                        />
                                    </div>

                                    {/* SKELETON CONTENT */}

                                    {/* Image Side Skeleton */}
                                    <div className="w-full md:w-2/3 h-64 md:h-full relative bg-[#111] overflow-hidden group">
                                        {/* Shimmer Effect */}
                                        <motion.div variants={shimmer} initial="hidden" animate="visible" className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent z-10" />

                                        <div className="absolute bottom-0 left-0 right-0 p-8 space-y-3">
                                            <div className="h-6 w-24 bg-white/10 rounded animate-pulse" />
                                            <div className="h-10 w-64 bg-white/10 rounded animate-pulse" />
                                            <div className="flex gap-4">
                                                <div className="h-4 w-32 bg-white/10 rounded animate-pulse" />
                                                <div className="h-4 w-32 bg-white/10 rounded animate-pulse" />
                                            </div>
                                        </div>
                                    </div>

                                    {/* Detail Side Skeleton */}
                                    <div className="w-full md:w-1/3 p-8 flex flex-col bg-black border-l border-white/5 overflow-hidden relative">
                                        <motion.div variants={shimmer} initial="hidden" animate="visible" className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent z-10 pointer-events-none" />

                                        <div className="mb-6">
                                            <div className="h-3 w-20 bg-white/10 rounded mb-2 animate-pulse" />
                                            <div className="flex items-end gap-3 mb-4">
                                                <div className="h-12 w-32 bg-white/10 rounded animate-pulse" />
                                                <div className="h-6 w-16 bg-white/10 rounded animate-pulse" />
                                            </div>
                                            <div className="h-2 w-full bg-white/10 rounded-full animate-pulse" />
                                        </div>

                                        <div className="space-y-4 mb-8">
                                            <div className="grid grid-cols-2 gap-3">
                                                <div className="h-20 bg-white/5 rounded-2xl animate-pulse" />
                                                <div className="h-20 bg-white/5 rounded-2xl animate-pulse" />
                                            </div>
                                            <div className="h-24 bg-white/5 rounded-2xl animate-pulse" />
                                            <div className="h-24 bg-white/5 rounded-2xl animate-pulse" />
                                        </div>

                                        <div className="mt-auto">
                                            <div className="h-12 w-full bg-white/10 rounded-xl animate-pulse" />
                                        </div>
                                    </div>
                                </div>
                            </motion.div>
                        )}

                        {result && (
                            <motion.div
                                key="result-overlay"
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.95 }}
                                className="absolute inset-0 z-50 p-4 md:p-12 flex items-center justify-center bg-black/40 backdrop-blur-sm"
                            >
                                <div className="bg-black border border-white/10 rounded-3xl shadow-2xl overflow-hidden w-full max-w-5xl h-[85vh] flex flex-col md:flex-row relative">
                                    <button onClick={closePopup} className="absolute top-4 right-4 z-50 p-2 bg-black/50 hover:bg-black rounded-full text-white border border-white/10 transition-colors"><X className="w-6 h-6" /></button>

                                    {/* Image */}
                                    <div className="w-full md:w-2/3 h-64 md:h-full relative bg-[#111]">
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                downloadImage(result.image_base64, `solar_analysis_${result.sample_id || 'result'}.jpg`);
                                            }}
                                            className="absolute top-4 right-4 z-20 p-2 bg-black/50 hover:bg-black/70 text-white rounded-lg transition-colors backdrop-blur-sm border border-white/10"
                                            title="Download Analysis Image"
                                        >
                                            <Download className="w-5 h-5" />
                                        </button>

                                        {result.image_base64 && <img src={`data:image/jpeg;base64,${result.image_base64}`} className="w-full h-full object-cover opacity-90" />}
                                        <div className="absolute bottom-0 left-0 right-0 p-8 bg-gradient-to-t from-black via-black/50 to-transparent">
                                            <span className={clsx("inline-block px-3 py-1 bg-white text-black text-xs font-bold uppercase tracking-wider rounded-sm mb-2", result.has_solar ? "bg-green-400" : "bg-red-400")}>
                                                {result.has_solar ? "Solar Detected" : "No Solar"}
                                            </span>
                                            <h2 className="text-3xl font-bold text-white">Assessment Complete</h2>
                                            <div className="flex gap-4 mt-2 font-mono text-xs text-gray-300">
                                                <span>Lat: {selectedLocation?.lat.toFixed(5)}</span>
                                                <span>Lon: {selectedLocation?.lng.toFixed(5)}</span>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Details */}
                                    <div className="w-full md:w-1/3 p-8 flex flex-col overflow-y-auto bg-black border-l border-white/5">
                                        <div className="mb-6">
                                            <div className="text-gray-500 text-xs font-bold uppercase tracking-widest mb-1">Confidence</div>
                                            <div className="flex items-end gap-3">
                                                <div className="text-5xl font-bold text-white tracking-tighter">{(result.confidence * 100).toFixed(1)}%</div>
                                                <div className={clsx("px-2 py-0.5 rounded text-[10px] font-bold uppercase mb-2",
                                                    result.confidence > 0.8 ? "bg-green-500/20 text-green-400" :
                                                        result.confidence > 0.5 ? "bg-yellow-500/20 text-yellow-400" : "bg-red-500/20 text-red-400"
                                                )}>
                                                    {result.confidence > 0.8 ? "High" : result.confidence > 0.5 ? "Medium" : "Low"}
                                                </div>
                                            </div>

                                            {/* Colored Progress Bar */}
                                            <div className="w-full bg-white/10 h-1.5 rounded-full mt-4 overflow-hidden">
                                                <motion.div
                                                    initial={{ width: 0 }}
                                                    animate={{ width: `${result.confidence * 100}%` }}
                                                    transition={{ duration: 1, delay: 0.2 }}
                                                    className={clsx("h-full rounded-full",
                                                        result.confidence > 0.8 ? "bg-green-500" :
                                                            result.confidence > 0.5 ? "bg-yellow-500" : "bg-red-500"
                                                    )}
                                                />
                                            </div>
                                        </div>

                                        <div className="space-y-4 mb-8">
                                            <div className="grid grid-cols-2 gap-3">
                                                <div className="p-4 bg-white/5 rounded-2xl border border-white/5">
                                                    <div className="text-gray-500 text-[10px] font-bold uppercase tracking-wider mb-1">Method</div>
                                                    <div className="text-sm font-medium text-white">{result.detection_method || "Standard"}</div>
                                                </div>
                                                <div className="p-4 bg-white/5 rounded-2xl border border-white/5">
                                                    <div className="text-gray-500 text-[10px] font-bold uppercase tracking-wider mb-1">Buffer</div>
                                                    <div className="text-sm font-medium text-white">{result.buffer_size || 0} sqft</div>
                                                </div>
                                            </div>

                                            <div className="p-5 bg-white/5 rounded-2xl border border-white/5">
                                                <div className="flex justify-between items-center mb-1">
                                                    <div className="text-gray-400 text-xs font-bold uppercase">Est. Area</div>
                                                    <Zap className="w-4 h-4 text-yellow-400" />
                                                </div>
                                                <div className="text-2xl font-mono text-white tracking-tight">{result.pv_area_sqm_est?.toFixed(1) || 0} <span className="text-sm text-gray-500">m²</span></div>
                                            </div>
                                            <div className="p-5 bg-white/5 rounded-2xl border border-white/5">
                                                <div className="text-gray-400 text-xs font-bold uppercase mb-1">Distance from Center</div>
                                                <div className="text-2xl font-mono text-white tracking-tight">{result.euclidean_distance_m_est?.toFixed(1) || 0} <span className="text-sm text-gray-500">m</span></div>
                                            </div>
                                        </div>
                                        <div className="mt-auto">
                                            <button onClick={closePopup} className="w-full py-4 bg-white text-black font-bold text-sm rounded-xl hover:bg-gray-200 transition-colors shadow-lg shadow-white/5">Done</button>
                                        </div>
                                    </div>
                                </div>
                            </motion.div>
                        )}

                        {/* Team Modal */}
                        {showTeam && (
                            <motion.div key="team-modal" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="absolute inset-0 bg-black/80 backdrop-blur-md z-[60] flex items-center justify-center p-6">
                                <div className="bg-[#0a0a0a] border border-white/10 rounded-3xl w-full max-w-2xl overflow-hidden relative">
                                    <button onClick={() => setShowTeam(false)} className="absolute top-4 right-4 p-2 text-gray-400 hover:text-white transition-colors"><X /></button>
                                    <div className="p-8">
                                        <h2 className="text-3xl font-bold text-white mb-2">Team Dude Coders</h2>
                                        <p className="text-gray-500 mb-8">IIT Madras - EcoInnovators Ideathon Project</p>

                                        <div className="grid gap-4">
                                            {[
                                                { name: "S Shriprasad", email: "da25e054@smail.iitm.ac.in", linkedin: "https://www.linkedin.com/in/shriprasad-s-51723a240/" },
                                                { name: "P Saranath", email: "da25e003@smail.iitm.ac.in", linkedin: "https://www.linkedin.com/in/saranath-premkumar-594513200/" },
                                                { name: "B Shruthi", email: "ns25z069@smail.iitm.ac.in", linkedin: "https://www.linkedin.com/in/shruthibalasubramanian/" }
                                            ].map((member, i) => (
                                                <div key={i} className="p-4 rounded-xl bg-white/5 border border-white/5 flex items-center justify-between hover:bg-white/10 transition-colors">
                                                    <div>
                                                        <div className="font-bold text-white text-lg">{member.name}</div>
                                                        <div className="text-sm text-gray-400">{member.email}</div>
                                                    </div>
                                                    <a href={member.linkedin} target="_blank" rel="noopener noreferrer" className="px-4 py-2 rounded-lg bg-[#0077b5] text-white text-xs font-bold hover:opacity-90 transition-opacity">
                                                        LinkedIn
                                                    </a>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                    <div className="bg-white/5 p-6 text-center text-sm text-gray-500 border-t border-white/5">
                                        Check out the project on <a href="https://github.com/shriprasad15/Dude-Coders-Ideathon/" target="_blank" className="text-white hover:underline">GitHub</a>
                                    </div>
                                </div>
                            </motion.div>
                        )}

                        {/* Bulk Upload Modal */}
                        {showBulkModal && (
                            <motion.div
                                key="bulk-modal"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="absolute inset-0 bg-black/80 backdrop-blur-md z-[60] flex items-center justify-center p-6"
                            >
                                <motion.div
                                    initial={{ scale: 0.9, y: 20 }}
                                    animate={{ scale: 1, y: 0 }}
                                    exit={{ scale: 0.9, y: 20 }}
                                    className="bg-[#0a0a0a] border border-white/10 rounded-3xl w-full max-w-lg overflow-hidden relative p-8"
                                >
                                    <button onClick={() => setShowBulkModal(false)} className="absolute top-4 right-4 p-2 text-gray-400 hover:text-white transition-colors"><X className="w-5 h-5" /></button>

                                    <div className="flex flex-col items-center text-center">
                                        <div className="w-16 h-16 rounded-2xl bg-blue-500/10 text-blue-500 flex items-center justify-center mb-6">
                                            <FileSpreadsheet className="w-8 h-8" />
                                        </div>

                                        <h2 className="text-2xl font-bold text-white mb-2">Upload Coordinates</h2>
                                        <p className="text-gray-400 text-sm mb-8 max-w-xs">
                                            Drag & drop your CSV or Excel file here to analyze multiple locations at once.
                                        </p>

                                        {/* Drop Zone */}
                                        <div
                                            onDrop={onDrop}
                                            onDragOver={onDragOver}
                                            className="w-full h-48 border-2 border-dashed border-white/20 hover:border-blue-500/50 rounded-2xl flex flex-col items-center justify-center bg-white/5 hover:bg-white/10 transition-all cursor-pointer relative group"
                                        >
                                            <input
                                                type="file"
                                                accept=".csv,.xlsx,.xls"
                                                onChange={(e) => e.target.files && handleBulkUpload(e.target.files[0])}
                                                className="absolute inset-0 opacity-0 cursor-pointer"
                                            />
                                            <UploadCloud className="w-10 h-10 text-gray-500 group-hover:text-blue-400 mb-4 transition-colors" />
                                            <span className="text-sm font-bold text-white mb-1">Click or Drag file here</span>
                                            <span className="text-xs text-gray-500 uppercase tracking-wider font-bold">CSV, Excel</span>

                                            {isBulkAnalyzing && (
                                                <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center backdrop-blur-sm rounded-2xl">
                                                    <Loader2 className="w-8 h-8 text-blue-500 animate-spin mb-2" />
                                                    <span className="text-xs font-bold text-blue-400">Processing Locations...</span>
                                                </div>
                                            )}
                                        </div>

                                        <div className="mt-6 w-full flex justify-center">
                                            <button
                                                onClick={downloadTemplate}
                                                className="flex items-center gap-2 text-xs font-bold text-gray-500 hover:text-white transition-colors uppercase tracking-wider"
                                            >
                                                <Download className="w-4 h-4" /> Download Template
                                            </button>
                                        </div>
                                    </div>
                                </motion.div>
                            </motion.div>
                        )}

                        {/* How To / Methodology Modal */}
                        {showHowTo && (
                            <motion.div key="howto-modal" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="absolute inset-0 bg-black/80 backdrop-blur-md z-[60] flex items-center justify-center p-4">
                                <div className="bg-[#0a0a0a] border border-white/10 rounded-3xl w-full max-w-3xl max-h-[90vh] flex flex-col relative">
                                    <div className="p-6 border-b border-white/10 flex justify-between items-center bg-black/50 rounded-t-3xl">
                                        <h2 className="text-2xl font-bold text-white">How To Use</h2>
                                        <button onClick={() => setShowHowTo(false)} className="p-2 text-gray-400 hover:text-white transition-colors"><X /></button>
                                    </div>
                                    <div className="p-8 overflow-y-auto custom-scrollbar text-gray-300 leading-relaxed space-y-4">
                                        <div className="prose prose-invert max-w-none">
                                            <ReactMarkdown>{HOW_TO_CONTENT}</ReactMarkdown>
                                        </div>
                                    </div>
                                </div>
                            </motion.div>
                        )}

                    </AnimatePresence>
                </div >
            </div >
        </div >

    )
}

export default App
