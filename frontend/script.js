const templateInput = document.getElementById('template-input');
const testInput = document.getElementById('test-input');
const templatePreview = document.getElementById('template-preview');
const testPreview = document.getElementById('test-preview');
const analyzeBtn = document.getElementById('analyze-btn');
const resultsSection = document.getElementById('results-section');
const resultCanvas = document.getElementById('result-canvas');
const defectsList = document.getElementById('defects-list');
const exportBtn = document.getElementById('export-btn');
const modeRadios = document.getElementsByName('mode');
const templateZone = document.getElementById('template-card');

let templateFile = null;
let testFile = null;
let currentDefects = [];

// Setup Drop Zones
setupDropZone('drop-zone-template', templateInput);
setupDropZone('drop-zone-test', testInput);

function setupDropZone(id, input) {
    const zone = document.getElementById(id);
    
    zone.addEventListener('click', () => input.click());
    
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.style.borderColor = 'var(--accent-color)';
    });

    zone.addEventListener('dragleave', () => {
        zone.style.borderColor = 'var(--glass-border)';
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.style.borderColor = 'var(--glass-border)';
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            handleFileSelect(input);
        }
    });

    input.addEventListener('change', () => handleFileSelect(input));
}

function handleFileSelect(input) {
    const file = input.files[0];
    if (!file) return;

    const isTemplate = input.id === 'template-input';
    const preview = isTemplate ? templatePreview : testPreview;
    const nameSpan = document.getElementById(isTemplate ? 'template-name' : 'test-name');

    if (isTemplate) templateFile = file;
    else testFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.hidden = false;
        nameSpan.textContent = file.name;
    };
    reader.readAsDataURL(file);

    updateAnalyzeButton();
}

// Mode Switching Logic
modeRadios.forEach(radio => {
    radio.addEventListener('change', (e) => {
        if (e.target.value === 'ai') {
            templateZone.style.display = 'none';
        } else {
            templateZone.style.display = 'flex';
        }
        updateAnalyzeButton();
    });
});

function updateAnalyzeButton() {
    const mode = document.querySelector('input[name="mode"]:checked').value;
    if (mode === 'ai') {
        analyzeBtn.disabled = !testFile;
    } else {
        analyzeBtn.disabled = !(templateFile && testFile);
    }
}

analyzeBtn.addEventListener('click', async () => {
    const mode = document.querySelector('input[name="mode"]:checked').value;
    if (mode === 'reference' && (!templateFile || !testFile)) return;
    if (mode === 'ai' && !testFile) return;

    // UI Loading State
    analyzeBtn.disabled = true;
    analyzeBtn.querySelector('.btn-text').textContent = 'Processing...';
    analyzeBtn.querySelector('.loader').hidden = false;
    resultsSection.hidden = true;

    const formData = new FormData();
    formData.append('test', testFile);
    formData.append('mode', mode);
    
    if (mode === 'reference') {
        formData.append('template', templateFile);
    }

    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            displayResults(data.defects);
        }

    } catch (error) {
        console.error('Error:', error);
        alert('Failed to connect to backend. Is it running?');
    } finally {
        // Reset Button
        analyzeBtn.disabled = false;
        analyzeBtn.querySelector('.btn-text').textContent = 'Analyze Circuit';
        analyzeBtn.querySelector('.loader').hidden = true;
    }
});

function displayResults(defects) {
    currentDefects = defects;
    resultsSection.hidden = false;
    defectsList.innerHTML = '';

    // Draw on Canvas
    const ctx = resultCanvas.getContext('2d');
    const img = new Image();
    img.src = testPreview.src;
    
    img.onload = () => {
        // Set canvas size to match image
        resultCanvas.width = img.naturalWidth;
        resultCanvas.height = img.naturalHeight;
        
        // Draw image
        ctx.drawImage(img, 0, 0);

        // Draw boxes
        defects.forEach(defect => {
            const [x, y, w, h] = defect.bbox;
            
            // 1. Semi-transparent fill
            ctx.fillStyle = 'rgba(239, 68, 68, 0.15)';
            ctx.fillRect(x, y, w, h);

            // 2. Border
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);

            // 3. Label Tag
            const text = `${defect.label} ${(defect.confidence * 100).toFixed(0)}%`;
            ctx.font = '600 16px Outfit'; // Use new font
            const textMetrics = ctx.measureText(text);
            const textWidth = textMetrics.width;
            const textHeight = 24;

            // Label Background
            ctx.fillStyle = '#ef4444';
            ctx.beginPath();
            ctx.roundRect(x, y - textHeight - 4, textWidth + 16, textHeight, [4, 4, 4, 4]);
            ctx.fill();

            // Label Text
            ctx.fillStyle = '#ffffff';
            ctx.fillText(text, x + 8, y - 8);

            // Add to list
            addDefectToList(defect);
        });
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    };
}

function addDefectToList(defect) {
    const div = document.createElement('div');
    div.className = 'defect-item';
    div.innerHTML = `
        <span class="defect-label">${defect.label}</span>
        <span class="defect-conf">Confidence: ${(defect.confidence * 100).toFixed(1)}%</span>
    `;
    defectsList.appendChild(div);
}

exportBtn.addEventListener('click', () => {
    // 1. Download Image
    const link = document.createElement('a');
    link.download = 'circuitguard_analysis.png';
    link.href = resultCanvas.toDataURL();
    link.click();

    // 2. Download CSV
    if (currentDefects && currentDefects.length > 0) {
        let csvContent = "data:text/csv;charset=utf-8,Label,Confidence,X,Y,W,H\n";
        
        currentDefects.forEach(d => {
            const [x, y, w, h] = d.bbox;
            const row = `${d.label},${(d.confidence * 100).toFixed(2)}%,${x},${y},${w},${h}`;
            csvContent += row + "\n";
        });

        const encodedUri = encodeURI(csvContent);
        const csvLink = document.createElement("a");
        csvLink.setAttribute("href", encodedUri);
        csvLink.setAttribute("download", "circuitguard_logs.csv");
        document.body.appendChild(csvLink);
        csvLink.click();
        document.body.removeChild(csvLink);
    }
});
