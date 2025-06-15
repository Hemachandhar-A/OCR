import React, { useState, useEffect } from 'react';

// Main App component
function App() {
  const [selectedImage, setSelectedImage] = useState(null); // The original image file
  const [highlightedImageSrc, setHighlightedImageSrc] = useState(null); // Base64 for the processed image
  const [textRegionsDetected, setTextRegionsDetected] = useState(0); // Count from backend
  const [isDragOver, setIsDragOver] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  // ocrResult will now hold a message about future enhancement
  const [ocrFeatureMessage, setOcrFeatureMessage] = useState("Text extraction coming soon! For now, you can see the detected text highlighted in the image preview.");
  const [showResultPage, setShowResultPage] = useState(false);
  const [error, setError] = useState('');
  const [messageBox, setMessageBox] = useState({ visible: false, content: '' });
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Define your backend API URL
  const API_BASE_URL = 'https://ocr-production-0ada.up.railway.app'; // Make sure this matches your Flask backend URL

  // Function to show custom message box
  const showCustomMessageBox = (message) => {
    setMessageBox({ visible: true, content: message });
    setTimeout(() => {
      setMessageBox({ visible: false, content: '' });
    }, 3000);
  };

  // Handler for file input change
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Allow only image files (JPG, PNG) as PDF processing for text detection is not explicitly supported by the current backend
      // Your backend only expects image input for text detection model
      if (file.type.startsWith('image/')) {
        setSelectedImage(file);
        setError('');
      } else {
        setError('Please upload an image file (JPG, PNG). PDF support for detection is not yet available.');
        setSelectedImage(null);
      }
    }
  };

  // Handlers for drag and drop functionality
  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragOver(false);
    const file = event.dataTransfer.files[0];
    if (file) {
      // Allow only image files for the same reason as above
      if (file.type.startsWith('image/')) {
        setSelectedImage(file);
        setError('');
      } else {
        setError('Please drop an image file (JPG, PNG).');
        setSelectedImage(null);
      }
    }
  };

  // Text detection processing function (renamed from processOCR for clarity)
  const processTextDetection = async () => {
    if (!selectedImage) {
      setError('Please select an image to process.');
      return;
    }

    setIsProcessing(true);
    setError('');
    setHighlightedImageSrc(null); // Clear previous result
    setTextRegionsDetected(0); // Clear previous count

    console.log('Sending image to backend for text detection:', selectedImage.name);

    const formData = new FormData();
    formData.append('file', selectedImage); // 'file' must match the backend's expected field name

    try {
      const response = await fetch(`${API_BASE_URL}/detect-text`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        // Attempt to read JSON error first, then fall back to status text
        let errorData = {};
        try {
            errorData = await response.json();
        } catch (jsonError) {
            // If JSON parsing fails, just use the status text
            errorData = { error: response.statusText };
        }
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('Backend response:', result);

      if (result.success) {
        setHighlightedImageSrc(result.highlighted_image);
        setTextRegionsDetected(result.text_regions_detected);
        showCustomMessageBox('🎉 Text detection complete!');
        setShowResultPage(true);
      } else {
        setError(result.error || 'Unknown error occurred during detection.');
        showCustomMessageBox('❌ Detection failed. Please try again.');
      }
    } catch (err) {
      console.error("Text detection failed:", err);
      setError(`Failed to connect to backend or process image: ${err.message}. Please ensure the backend is running.`);
      showCustomMessageBox('❌ Detection failed. Please check backend connection.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleBackToInput = () => {
    setShowResultPage(false);
    setSelectedImage(null);
    setHighlightedImageSrc(null);
    setTextRegionsDetected(0);
    setError('');
  };

  return (
    // Root container: min-h-screen for height, w-screen for full viewport width.
    // Ensure overflow-x-hidden is here to catch any rogue horizontal overflows.
    <div className="min-h-screen w-screen bg-gradient-to-br from-gray-900 via-slate-900 to-gray-800 text-gray-50 flex flex-col font-inter relative overflow-x-hidden">
      {/* Animated Background - These elements have negative margins/positions, 
          but overflow-x-hidden on the parent should contain them */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-teal-500/10 rounded-full blur-3xl animate-float"></div>
        <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-float-delayed"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-emerald-500/5 rounded-full blur-3xl animate-pulse-slow"></div>
      </div>

      {/* Header - Already full width */}
      <header className="w-full bg-gray-800/80 backdrop-blur-md border-b border-gray-700/50 sticky top-0 z-40 shadow-lg">
        <nav className="container mx-auto flex justify-between items-center px-4 sm:px-6 lg:px-8 py-4">
          <div className="text-2xl sm:text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-emerald-400 animate-gradient-x">
            OCRify
          </div>
          
          {/* Desktop Navigation */}
          <div className="hidden md:flex space-x-8">
            {['Home', 'About', 'How It Works', 'Contact'].map((item, index) => (
              <a 
                key={item}
                href="#" 
                className="text-gray-300 hover:text-teal-300 transition-all duration-300 hover:scale-105 relative group"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                {item}
                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-teal-400 to-emerald-400 transition-all duration-300 group-hover:w-full"></span>
              </a>
            ))}
          </div>
          
          {/* Mobile Menu Button */}
          <button 
            className="md:hidden text-gray-300 hover:text-teal-300 focus:outline-none transition-all duration-300 hover:scale-110"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            <svg className={`w-6 h-6 transition-transform duration-300 ${mobileMenuOpen ? 'rotate-90' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d={mobileMenuOpen ? "M6 18L18 6M6 6l12 12" : "M4 6h16M4 12h16m-7 6h7"}></path>
            </svg>
          </button>
        </nav>
        
        {/* Mobile Menu */}
        <div className={`md:hidden overflow-hidden transition-all duration-300 ease-in-out ${mobileMenuOpen ? 'max-h-64 opacity-100' : 'max-h-0 opacity-0'}`}>
          <div className="px-4 pt-2 pb-4 space-y-2 bg-gray-800/90 backdrop-blur-sm">
            {['Home', 'About', 'How It Works', 'Contact'].map((item, index) => (
              <a 
                key={item}
                href="#" 
                className="block px-3 py-2 text-gray-300 hover:text-teal-300 hover:bg-gray-700/50 rounded-lg transition-all duration-300 transform hover:translate-x-2"
                style={{ animationDelay: `${index * 50}ms` }}
              >
                {item}
              </a>
            ))}
          </div>
        </div>
      </header>

      {/* Main Content Area - Stretches full width, padding only inside for content centering */}
      <main className="flex-grow w-full py-8 sm:py-12 flex flex-col items-center justify-center relative z-10">
        {showResultPage ? (
          // Detection Result Page - container mx-auto for content width and horizontal padding
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:py-12 bg-gray-800/80 backdrop-blur-md rounded-2xl shadow-2xl border border-gray-700/50 animate-slide-up">
            <h2 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-center text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-emerald-400 mb-8 animate-fade-in">
              ✨ Text Detection Results
            </h2>
            
            <div className="flex flex-col xl:flex-row gap-8 lg:gap-12 items-start"> {/* items-start for better top alignment */}
              {/* Highlighted Image Preview Section */}
              {highlightedImageSrc && (
                <div className="xl:w-1/2 w-full flex flex-col items-center justify-center p-6 sm:p-8 border border-gray-700/50 rounded-xl bg-gradient-to-br from-gray-700/30 to-gray-800/30 backdrop-blur-sm animate-fade-in-left">
                  <h3 className="text-lg sm:text-xl font-semibold mb-4 text-gray-200 text-center">
                    📸 Detected Text Highlighted
                  </h3>
                  <div className="relative group">
                    <img
                      src={highlightedImageSrc}
                      alt="Highlighted Text"
                      className="max-w-full h-auto max-h-[400px] lg:max-h-[350px] xl:max-h-[400px] rounded-lg shadow-xl object-contain transition-transform duration-300 group-hover:scale-105" // object-contain ensures image fits
                      onError={(e) => { e.target.onerror = null; e.target.src = 'https://placehold.co/400x300/374151/FFFFFF?text=Image+Load+Error'; }}
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                  </div>
                  <div className="mt-4 px-3 py-1 bg-gray-600/50 rounded-full text-gray-300 text-xs sm:text-sm truncate max-w-full">
                    {selectedImage ? selectedImage.name : 'Processed Image'}
                  </div>
                  <p className="mt-2 text-sm sm:text-base text-teal-300 font-medium">
                    {textRegionsDetected > 0 ? `Areas with text detected: ${textRegionsDetected} pixels` : 'No text regions detected (or very few).'}
                  </p>
                </div>
              )}
              
              {/* Future Enhancement Text Section */}
              <div className="xl:w-1/2 w-full animate-fade-in-right">
                <h3 className="text-lg sm:text-xl font-semibold mb-4 text-gray-200 flex items-center">
                  <span className="mr-2">📝</span> Extracted Text (Future Enhancement)
                </h3>
                <textarea
                  className="w-full h-72 sm:h-80 lg:h-[400px] xl:h-[400px] p-4 bg-gray-700/50 backdrop-blur-sm border border-gray-600/50 rounded-xl text-gray-100 placeholder-gray-400 focus:ring-2 focus:ring-teal-500 focus:border-teal-500 resize-none transition-all duration-300 shadow-inner leading-relaxed text-base"
                  value={ocrFeatureMessage}
                  readOnly
                  aria-label="Extracted text result"
                  placeholder="Extracted text will appear here..."
                ></textarea>
                
                {/* Action Buttons - disabled or show "coming soon" */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 mt-6">
                  <button
                    onClick={() => showCustomMessageBox('📋 Text extraction functionality coming soon!')}
                    disabled={true} // Disable as feature is not available
                    className="px-4 py-3 bg-gradient-to-r from-teal-600 to-teal-700 text-white font-semibold rounded-xl shadow-lg focus:outline-none focus:ring-2 focus:ring-teal-500 focus:ring-offset-2 focus:ring-offset-gray-900 transition-all duration-300 flex items-center justify-center group transform opacity-60 cursor-not-allowed lg:px-3 lg:py-2 lg:text-sm"
                  >
                    <svg className="w-4 h-4 sm:w-5 sm:h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"></path>
                    </svg>
                    <span className="hidden sm:inline">Copy (Soon)</span>
                  </button>
                  
                  <button
                    onClick={() => showCustomMessageBox('✏️ Edit functionality coming soon!')}
                    disabled={true} // Disable as feature is not available
                    className="px-4 py-3 bg-gradient-to-r from-blue-600 to-blue-700 text-white font-semibold rounded-xl shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 transition-all duration-300 flex items-center justify-center group transform opacity-60 cursor-not-allowed lg:px-3 lg:py-2 lg:text-sm"
                  >
                    <svg className="w-4 h-4 sm:w-5 sm:h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.232z"></path>
                    </svg>
                    <span className="hidden sm:inline">Edit (Soon)</span>
                  </button>
                  
                  <button
                    onClick={() => showCustomMessageBox('💾 Download functionality coming soon!')}
                    disabled={true} // Disable as feature is not available
                    className="px-4 py-3 bg-gradient-to-r from-purple-600 to-purple-700 text-white font-semibold rounded-xl shadow-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 focus:ring-offset-gray-900 transition-all duration-300 flex items-center justify-center group transform opacity-60 cursor-not-allowed lg:px-3 lg:py-2 lg:text-sm"
                  >
                    <svg className="w-4 h-4 sm:w-5 sm:h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                    </svg>
                    <span className="hidden sm:inline">Save (Soon)</span>
                  </button>
                  
                  <button
                    onClick={handleBackToInput}
                    className="px-4 py-3 bg-gradient-to-r from-gray-600 to-gray-700 text-white font-semibold rounded-xl shadow-lg hover:from-gray-700 hover:to-gray-800 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 focus:ring-offset-gray-900 transition-all duration-300 flex items-center justify-center group transform hover:scale-105 hover:-translate-y-1 lg:px-3 lg:py-2 lg:text-sm"
                  >
                    <svg className="w-4 h-4 sm:w-5 sm:h-5 mr-2 group-hover:-translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 15l-3-3m0 0l3-3m-3 3h8M3 12a9 9 0 1118 0 9 9 0 01-18 0z"></path>
                    </svg>
                    <span className="hidden sm:inline">New</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : (
          // OCR Input Page container - container mx-auto for content width and horizontal padding
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:py-12 bg-gray-800/80 backdrop-blur-md rounded-2xl shadow-2xl text-center border border-gray-700/50 animate-slide-up">
            <div className="mb-8 sm:mb-12">
              <h1 className="text-3xl sm:text-4xl lg:text-5xl xl:text-6xl 2xl:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-teal-400 via-emerald-400 to-cyan-400 mb-4 sm:mb-6 animate-gradient-x">
                🚀 OCRify Your Documents
              </h1>
              <p className="text-lg sm:text-xl lg:text-xl xl:text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed animate-fade-in-up">
                Transform images into detected text highlights with our powerful AI. Text extraction coming soon!
              </p>
            </div>

            {/* Drag and Drop Area */}
            <div
              className={`border-2 border-dashed rounded-2xl p-8 sm:p-12 lg:p-12 xl:p-12 2xl:p-10 cursor-pointer transition-all duration-500 ease-in-out mb-6 sm:mb-8 flex flex-col items-center justify-center relative overflow-hidden group ${
                isDragOver 
                  ? 'border-teal-400 bg-teal-500/10 scale-102 shadow-xl shadow-teal-500/20' 
                  : 'border-gray-600 hover:border-teal-400 hover:bg-gray-700/20'
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              {/* Animated background effect */}
              <div className="absolute inset-0 bg-gradient-to-r from-teal-500/5 via-transparent to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              
              <input
                type="file"
                accept="image/*" // Restrict to image types only
                onChange={handleFileChange}
                className="hidden"
                id="file-upload"
              />
              <label htmlFor="file-upload" className="flex flex-col items-center justify-center w-full h-full text-center relative z-10">
                <div className="relative">
                  <svg className={`w-16 h-16 sm:w-20 sm:h-20 lg:w-20 lg:h-20 xl:w-20 xl:h-20 text-teal-400 mb-6 transition-all duration-500 ${isDragOver ? 'animate-bounce scale-110' : 'animate-float'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                  </svg>
                  {isDragOver && (
                    <div className="absolute inset-0 bg-teal-400/20 rounded-full animate-ping"></div>
                  )}
                </div>
                
                <p className="text-xl sm:text-2xl lg:text-2xl xl:text-2xl font-bold text-gray-200 mb-2 animate-fade-in">
                  {isDragOver ? '🎯 Drop it here!' : '📁 Drag & Drop your image'}
                </p>
                <p className="text-sm sm:text-base lg:text-base xl:text-base text-gray-400 mb-6 animate-fade-in-delay">
                  Support for JPG, PNG files
                </p>
                {/* Adjusted "or" to align better */}
                <div className="relative my-4 w-full max-w-xs">
                    <div className="absolute inset-0 flex items-center" aria-hidden="true">
                      <div className="w-full border-t border-gray-600"></div>
                    </div>
                    <div className="relative flex justify-center text-sm">
                      <span className="px-2 bg-gray-800/80 text-gray-500">or</span>
                    </div>
                </div>
                
                <span className="inline-flex items-center px-6 sm:px-8 py-3 sm:py-4 bg-gradient-to-r from-teal-600 via-emerald-600 to-cyan-600 text-white font-bold rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105 hover:-translate-y-1 group-hover:animate-pulse xl:px-6 xl:py-3 xl:text-lg">
                  <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path>
                  </svg>
                  Browse Files
                </span>
              </label>
            </div>

            {/* Selected File Display */}
            {selectedImage && (
              <div className="mb-6 sm:mb-8 p-4 sm:p-6 xl:p-4 border border-gray-700/50 rounded-xl bg-gradient-to-r from-gray-700/30 to-gray-800/30 backdrop-blur-sm flex items-center justify-between animate-slide-in-right shadow-lg">
                <div className="flex items-center space-x-4 min-w-0 flex-1">
                  <div className="flex-shrink-0">
                    <svg className="w-8 h-8 xl:w-6 xl:h-6 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-gray-200 text-sm sm:text-lg xl:text-base font-medium truncate">
                      📎 <span className="font-semibold">{selectedImage.name}</span>
                    </p>
                    <p className="text-gray-400 text-xs sm:text-sm xl:text-xs">
                      {(selectedImage.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedImage(null)}
                  className="flex-shrink-0 text-red-400 hover:text-red-300 transition-all duration-200 hover:scale-110 p-2 rounded-full hover:bg-red-500/10"
                  aria-label="Remove selected image"
                >
                  <svg className="w-5 h-5 sm:w-6 sm:h-6 xl:w-5 xl:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
                  </svg>
                </button>
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-xl animate-shake">
                <p className="text-red-400 flex items-center justify-center">
                  <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                  {error}
                </p>
              </div>
            )}

            {/* Process Button */}
            <button
              onClick={processTextDetection} // Changed to new function name
              disabled={!selectedImage || isProcessing}
              className={`w-full py-4 sm:py-6 text-lg sm:text-xl xl:py-4 xl:text-xl font-bold rounded-xl shadow-2xl transition-all duration-500 transform relative overflow-hidden group ${
                !selectedImage || isProcessing
                  ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-teal-500 via-emerald-600 to-cyan-600 text-white hover:from-teal-600 hover:via-emerald-700 hover:to-cyan-700 hover:scale-105 hover:shadow-teal-500/25 active:scale-95'
              } flex items-center justify-center`}
            >
              {/* Button background animation */}
              {!isProcessing && selectedImage && (
                <div className="absolute inset-0 bg-gradient-to-r from-teal-400 via-emerald-500 to-cyan-500 opacity-0 group-hover:opacity-20 transition-opacity duration-500"></div>
              )}
              
              {isProcessing ? (
                <span className="flex items-center relative z-10">
                  <svg className="animate-spin -ml-1 mr-3 h-6 w-6 sm:h-6 sm:w-6 xl:h-6 xl:w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  🔄 Detecting Text...
                </span>
              ) : (
                <span className="flex items-center relative z-10">
                  <svg className="w-6 h-6 sm:h-6 sm:w-6 mr-3 group-hover:animate-bounce xl:w-6 xl:h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                  </svg>
                  ⚡ Start Text Detection
                </span>
              )}
            </button>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="w-full bg-gray-800/80 backdrop-blur-md border-t border-gray-700/50 p-4 sm:p-6 text-center text-gray-400 shadow-inner relative z-10">
        <div className="container mx-auto"> {/* Added container mx-auto for footer content centering */}
          <p className="text-sm sm:text-base mb-4">
            &copy; {new Date().getFullYear()} OCRify. Transforming documents with AI magic ✨
          </p>
          <div className="flex flex-wrap justify-center space-x-4 sm:space-x-8 text-xs sm:text-sm">
            <a href="#" className="hover:text-teal-400 transition-all duration-300 hover:scale-105 relative group">
              Privacy Policy
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-teal-400 transition-all duration-300 group-hover:w-full"></span>
            </a>
            <a href="#" className="hover:text-teal-400 transition-all duration-300 hover:scale-105 relative group">
              Terms of Service
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-teal-400 transition-all duration-300 group-hover:w-full"></span>
            </a>
            <a href="#" className="hover:text-teal-400 transition-all duration-300 hover:scale-105 relative group">
              Support
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-teal-400 transition-all duration-300 group-hover:w-full"></span>
            </a>
          </div>
        </div>
      </footer>

      {/* Custom Message Box */}
      {messageBox.visible && (
        <div className="fixed bottom-4 sm:bottom-6 left-1/2 transform -translate-x-1/2 bg-gradient-to-r from-teal-600 via-emerald-600 to-cyan-600 text-white px-6 py-4 rounded-xl shadow-2xl text-sm sm:text-lg font-semibold z-50 animate-bounce-in backdrop-blur-sm border border-white/20 max-w-sm mx-4">
          <div className="flex items-center justify-center">
            {messageBox.content}
          </div>
        </div>
      )}

      {/* Custom Styles */}
      <style jsx>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        /* Hide Scrollbar */
        /* For Webkit browsers (Chrome, Safari, Edge, Opera) */
        ::-webkit-scrollbar {
            width: 0px; /* For vertical scrollbar */
            height: 0px; /* For horizontal scrollbar */
            background: transparent; /* make scrollbar track transparent */
        }
        /* For Firefox */
        html, body {
            scrollbar-width: none; /* Firefox */
            -ms-overflow-style: none;  /* IE and Edge */
            overflow-x: hidden; /* Ensure no horizontal scrollbar */
        }
        /* For the root div of your app if it becomes the main scrollable area */
        .min-h-screen.w-screen {
            scrollbar-width: none; /* Firefox */
            -ms-overflow-style: none;  /* IE and Edge */
        }


        .font-inter {
          font-family: 'Inter', sans-serif;
        }

        /* Floating Animation */
        @keyframes float {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          33% { transform: translateY(-10px) rotate(1deg); }
          66% { transform: translateY(5px) rotate(-1deg); }
        }
        .animate-float {
          animation: float 6s ease-in-out infinite;
        }
        .animate-float-delayed {
          animation: float 8s ease-in-out infinite reverse;
        }

        /* Gradient Animation */
        @keyframes gradient-x {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        .animate-gradient-x {
          background-size: 200% 200%;
          animation: gradient-x 3s ease infinite;
        }

        /* Slide Animations */
        @keyframes slideUp {
          from { 
            opacity: 0; 
            transform: translateY(30px); 
          }
          to { 
            opacity: 1; 
            transform: translateY(0); 
          }
        }
        .animate-slide-up {
          animation: slideUp 0.6s ease-out forwards;
        }

        @keyframes slideInRight {
          from { 
            opacity: 0; 
            transform: translateX(30px); 
          }
          to { 
            opacity: 1; 
            transform: translateX(0); 
          }
        }
        .animate-slide-in-right {
          animation: slideInRight 0.5s ease-out forwards;
        }

        /* Fade Animations */
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        .animate-fade-in {
          animation: fadeIn 0.8s ease-out forwards;
        }

        @keyframes fadeInUp {
          from { 
            opacity: 0; 
            transform: translateY(20px); 
          }
          to { 
            opacity: 1; 
            transform: translateY(0); 
          }
        }
        .animate-fade-in-up {
          animation: fadeInUp 0.8s ease-out forwards;
        }

        .animate-fade-in-delay {
          animation: fadeInUp 0.8s ease-out 0.2s forwards;
          opacity: 0;
        }

        @keyframes fadeInLeft {
          from { 
            opacity: 0; 
            transform: translateX(-30px); 
          }
          to { 
            opacity: 1; 
            transform: translateX(0); 
          }
        }
        .animate-fade-in-left {
          animation: fadeInLeft 0.6s ease-out forwards;
        }

        @keyframes fadeInRight {
          from { 
            opacity: 0; 
            transform: translateX(30px); 
          }
          to { 
            opacity: 1; 
            transform: translateX(0); 
          }
        }
        .animate-fade-in-right {
          animation: fadeInRight 0.6s ease-out 0.2s forwards;
          opacity: 0;
        }

        /* Shake Animation */
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          25% { transform: translateX(-5px); }
          75% { transform: translateX(5px); }
        }
        .animate-shake {
          animation: shake 0.5s ease-in-out;
        }

        /* Bounce In Animation */
        @keyframes bounceIn {
          0% { 
            opacity: 0; 
            transform: translateX(-50%) scale(0.3) translateY(20px); 
          }
          50% { 
            opacity: 1; 
            transform: translateX(-50%) scale(1.05) translateY(-5px); 
          }
          70% { 
            transform: translateX(-50%) scale(0.95) translateY(0px); 
          }
          100% { 
            opacity: 1; 
            transform: translateX(-50%) scale(1) translateY(0px); 
          }
        }
        .animate-bounce-in {
          animation: bounceIn 0.6s ease-out forwards;
        }

        /* Pulse Slow */
        @keyframes pulseSlow {
          0%, 100% { opacity: 0.3; transform: scale(1); }
          50% { opacity: 0.6; transform: scale(1.05); }
        }
        .animate-pulse-slow {
          animation: pulseSlow 4s ease-in-out infinite;
        }

        /* Scale on Hover */
        .scale-102 {
          transform: scale(1.02);
        }

        /* Smooth Scrolling */
        html {
          scroll-behavior: smooth;
        }

        /* Responsive Grid */
        @media (max-width: 640px) {
          .grid-cols-2 {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
          }
        }

        @media (max-width: 480px) {
          .grid-cols-2 {
            grid-template-columns: 1fr;
            gap: 0.5rem;
          }
        }
      `}</style>
    </div>
  );
}

export default App;