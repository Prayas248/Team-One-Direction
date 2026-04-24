from flask import Flask, request, jsonify
from ingest import extract_text, chunk_text
from detect_lexical import detect_lexical
from detect_semantic import detect_semantic
from detect_intrinsic import detect_intrinsic
from score import aggregate_flags
from explain import explain_flag
import tempfile, os

app = Flask(__name__)
print("API script started...")
@app.route('/analyse', methods=['POST'])
def analyse():
    # Save the uploaded PDF temporarily
    f = request.files['pdf']
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        f.save(tmp.name)
    
    # Extract text and sections
    sections = extract_text(tmp.name)
    os.unlink(tmp.name)  # Delete the temporary file
    full_text = ' '.join(sections.values())
    chunks = chunk_text(full_text)

    # Run detection layers
    l1 = detect_lexical(chunks)
    l2 = detect_semantic(chunks)
    l3 = detect_intrinsic(sections)

    # Aggregate results
    result = aggregate_flags(l1, l2, l3)

    # Add explanations for High and Medium flags
    for flag in result['flags']:
        if flag['tier'] in ('HIGH', 'MEDIUM'):
            flag['explanation'] = explain_flag(flag)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)