const webcamElement = document.getElementById('webcam');
const captureBtn = document.getElementById('captureBtn');
const switchCameraBtn = document.getElementById('switchCameraBtn');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const resultsSection = document.getElementById('resultsSection');
const resultsMessage = document.getElementById('resultsMessage');
const successResults = document.getElementById('successResults');
const detectedBreedCard = document.getElementById('detectedBreedCard');
const breedCards = document.getElementById('breedCards');
const captureOverlay = document.getElementById('capture-overlay');

let net;
let currentFacingMode = 'environment';
let stream;

// Map some common ImageNet dog names to Dog CEO API names
const dogApiMap = {
    'chihuahua': 'chihuahua',
    'shih-tzu': 'shihtzu',
    'afghan hound': 'hound/afghan',
    'basset': 'hound/basset',
    'bloodhound': 'hound/blood',
    'english foxhound': 'hound/english',
    'beagle': 'beagle',
    'golden retriever': 'retriever/golden',
    'labrador retriever': 'retriever/chesapeake',
    'german shepherd': 'germanshepherd',
    'boxer': 'boxer',
    'pug': 'pug',
    'pomeranian': 'pomeranian',
    'chow': 'chow',
    'pembroke': 'corgi/cardigan',
    'cardigan': 'corgi/cardigan',
    'toy poodle': 'poodle/toy',
    'miniature poodle': 'poodle/miniature',
    'standard poodle': 'poodle/standard',
    'malamute': 'malamute',
    'siberian husky': 'husky',
    'dalmatian': 'dalmatian',
    'french bulldog': 'bulldog/french',
    'bull mastiff': 'mastiff/bull',
    'doberman': 'doberman',
    'rottweiler': 'rottweiler'
};

// Map of top breed -> list of similar breeds (using Dog API naming conventions or standard names)
const similarBreedsMap = {
    'doberman': ['rottweiler', 'pinscher', 'bloodhound'],
    'rottweiler': ['doberman', 'bull mastiff', 'boxer'],
    'golden retriever': ['labrador retriever', 'irish setter', 'cocker spaniel'],
    'labrador retriever': ['golden retriever', 'curly retriever', 'pointer'],
    'german shepherd': ['malinois', 'kelpie', 'malamute'],
    'pug': ['french bulldog', 'bull mastiff', 'boxer'],
    'siberian husky': ['malamute', 'akita', 'samoyed'],
    'chihuahua': ['papillon', 'toy terrier', 'miniature pinscher'],
    'beagle': ['basset hound', 'foxhound', 'dachshund'],
    'standard poodle': ['miniature poodle', 'spanish waterdog', 'bichon'],
    'french bulldog': ['pug', 'boston bulldog', 'bull mastiff'],
    'boxer': ['bull mastiff', 'french bulldog', 'pug'],
    'dalmatian': ['pointer', 'english hound', 'english setter'],
    'malamute': ['siberian husky', 'akita', 'samoyed'],
    'shih-tzu': ['lhasa', 'pekingese', 'maltese'],
    'pomeranian': ['japanese spitz', 'papillon', 'chihuahua'],
    'chow': ['akita', 'shiba', 'tibetan mastiff'],
    'corgi': ['dachshund', 'beagle', 'basset']
};

async function setupWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: currentFacingMode }
        });
        webcamElement.srcObject = stream;
        
        return new Promise((resolve) => {
            webcamElement.onloadedmetadata = () => {
                resolve();
            };
        });
    } catch (e) {
        console.error("Camera error:", e);
        alert("Could not access the camera. Please ensure you have granted permission.");
    }
}

async function loadModel() {
    try {
        net = await mobilenet.load({version: 2, alpha: 1.0});
        captureBtn.disabled = false;
        captureBtn.innerHTML = '<span class="btn-text">Capture & Analyze</span>';
    } catch (e) {
        console.error("Model loading failed:", e);
        captureBtn.innerHTML = '<span class="btn-text">Model Load Failed</span>';
    }
}

async function init() {
    await setupWebcam();
    await loadModel();
}

switchCameraBtn.addEventListener('click', async () => {
    currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
    await setupWebcam();
});

captureBtn.addEventListener('click', async () => {
    // Flash effect
    captureOverlay.classList.add('flash');
    setTimeout(() => captureOverlay.classList.remove('flash'), 100);
    
    captureBtn.disabled = true;
    captureBtn.innerHTML = '<div class="loading-spinner"></div>Analyzing...';
    
    canvas.width = webcamElement.videoWidth;
    canvas.height = webcamElement.videoHeight;
    ctx.drawImage(webcamElement, 0, 0, canvas.width, canvas.height);
    const capturedImageUrl = canvas.toDataURL('image/jpeg');
    
    try {
        const predictions = await net.classify(canvas, 10);
        
        const dogKeywords = ['dog', 'hound', 'terrier', 'spaniel', 'retriever', 'poodle', 'pug', 'chihuahua', 'husky', 'mastiff', 'bulldog', 'collie', 'shepherd', 'corgi', 'malamute', 'dalmatian', 'shih-tzu', 'doberman', 'rottweiler', 'pinscher', 'setter', 'pointer'];
        
        const isDogBreed = (className) => {
            const lowerName = className.toLowerCase();
            return dogKeywords.some(keyword => lowerName.includes(keyword));
        };
        
        const dogPredictions = predictions.filter(p => isDogBreed(p.className));
        
        displayResults(dogPredictions, predictions[0], capturedImageUrl);
    } catch (e) {
        console.error(e);
        alert("Error during classification.");
    } finally {
        captureBtn.disabled = false;
        captureBtn.innerHTML = '<span class="btn-text">Capture & Analyze</span>';
    }
});

async function displayResults(dogPredictions, topPrediction, capturedImageUrl) {
    resultsSection.style.display = 'block';
    
    if (dogPredictions.length === 0) {
        successResults.style.display = 'none';
        resultsMessage.style.display = 'block';
        resultsMessage.textContent = `No dog detected. Looks more like a ${topPrediction.className.split(',')[0]} (${(topPrediction.probability*100).toFixed(1)}%).`;
        return;
    }
    
    resultsMessage.style.display = 'none';
    successResults.style.display = 'block';
    
    // Top breed logic
    const primaryPrediction = dogPredictions[0];
    const topBreedName = primaryPrediction.className.split(',')[0].trim().toLowerCase();
    
    // Display detected breed card
    detectedBreedCard.innerHTML = `
        <img src="${capturedImageUrl}" class="detected-breed-card-img" alt="Captured Image">
        <div class="detected-breed-card-content">
            <div class="detected-breed-name">${capitalize(topBreedName)}</div>
            <div class="detected-breed-confidence">${(primaryPrediction.probability * 100).toFixed(1)}% Confidence</div>
        </div>
    `;
    
    // Determine similar breeds
    let similarBreedsToFetch = [];
    
    // Check if we have predefined similar breeds
    const exactMatchKey = Object.keys(similarBreedsMap).find(k => topBreedName.includes(k));
    
    if (exactMatchKey) {
        similarBreedsToFetch = similarBreedsMap[exactMatchKey];
    } else {
        // Fallback: Use the next best predictions from the model itself as they look similar
        for (let i = 1; i < Math.min(4, dogPredictions.length); i++) {
            similarBreedsToFetch.push(dogPredictions[i].className.split(',')[0].trim().toLowerCase());
        }
    }
    
    // Render similar breeds
    breedCards.innerHTML = '';
    
    for (const breedName of similarBreedsToFetch) {
        const apiBreedPath = getDogApiPath(breedName);
        let imageUrl = '';
        
        if (apiBreedPath) {
            try {
                const res = await fetch(`https://dog.ceo/api/breed/${apiBreedPath}/images/random`);
                const data = await res.json();
                if (data.status === "success") {
                    imageUrl = data.message;
                }
            } catch (e) {
                console.error("Failed to fetch image for", apiBreedPath);
            }
        }
        
        // Fallback image
        if (!imageUrl) {
            imageUrl = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="%23A68B7C" stroke-width="2"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/><path d="M12 6c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6-2.69-6-6-6zm0 10c-2.21 0-4-1.79-4-4s1.79-4 4-4 4 1.79 4 4-1.79 4-4 4z"/></svg>';
        }
        
        const card = document.createElement('div');
        card.className = 'breed-card';
        card.innerHTML = `
            <img src="${imageUrl}" alt="${breedName}" class="breed-card-img" onerror="this.style.display='none'">
            <div class="breed-card-content">
                <div class="breed-name">${capitalize(breedName)}</div>
            </div>
        `;
        breedCards.appendChild(card);
    }
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function getDogApiPath(className) {
    const lower = className.toLowerCase();
    if (dogApiMap[lower]) return dogApiMap[lower];
    if (lower.includes('hound')) return 'hound';
    if (lower.includes('retriever')) return 'retriever';
    if (lower.includes('terrier')) return 'terrier';
    if (lower.includes('spaniel')) return 'spaniel';
    if (lower.includes('poodle')) return 'poodle';
    if (lower.includes('pug')) return 'pug';
    if (lower.includes('husky')) return 'husky';
    if (lower.includes('bulldog')) return 'bulldog';
    if (lower.includes('collie')) return 'collie';
    if (lower.includes('corgi')) return 'corgi/cardigan';
    if (lower.includes('doberman')) return 'doberman';
    if (lower.includes('rottweiler')) return 'rottweiler';
    if (lower.includes('pinscher')) return 'pinscher';
    if (lower.includes('setter')) return 'setter';
    
    return lower.split(' ')[0];
}

function capitalize(str) {
    return str.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
}

init();
