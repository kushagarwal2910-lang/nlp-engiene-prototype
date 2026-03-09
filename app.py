import spacy
import torch
from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from langdetect import detect as lang_detect

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
rb_tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
rb_model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english", output_attentions=True)
rb_model.eval()

FRAMEWORKS = {
    "Heuristics": [
        "The system should always keep users informed about what is going on.",
        "The system should speak the users language, using words familiar to the user.",
        "Users need a clearly marked emergency exit to leave unwanted states.",
        "Users should not wonder whether different words or actions mean the same thing.",
        "Careful design prevents problems from occurring in the first place.",
        "Minimize the users memory load by making options visible.",
        "The system should cater to both inexperienced and experienced users.",
        "Dialogues should not contain irrelevant or rarely needed information.",
        "Error messages should indicate the problem and suggest a solution.",
        "Help and documentation should be easy to search and focused on the task."
    ],
    "SUS": [
        "I would like to use this system frequently.",
        "I found the system unnecessarily complex.",
        "I thought the system was easy to use.",
        "I would need technical support to use this system.",
        "The functions in this system were well integrated.",
        "There was too much inconsistency in this system.",
        "Most people would learn to use this system very quickly.",
        "The system was very cumbersome to use.",
        "I felt very confident using the system.",
        "I needed to learn a lot before I could use this system."
    ],
    "NASA-TLX": [
        "How much mental and perceptual activity was required for the task.",
        "How much physical activity was required for the task.",
        "How much time pressure was felt during the task.",
        "How successful were you in accomplishing the task goals.",
        "How hard did you have to work to accomplish your performance level.",
        "How insecure, discouraged, irritated or stressed did you feel during the task."
    ],
    "User Testing": [
        "Whether the user successfully completed the assigned task.",
        "The time and steps taken to complete the task.",
        "The number of errors made while attempting tasks.",
        "How quickly a new user can learn to use the interface.",
        "The subjective satisfaction and enjoyment of the user.",
        "How the user navigates and moves through the interface.",
        "Spoken comments and think-aloud observations during testing."
    ]
}

FRAMEWORK_LABELS = {
    "Heuristics": ["H1","H2","H3","H4","H5","H6","H7","H8","H9","H10"],
    "SUS": ["SUS1","SUS2","SUS3","SUS4","SUS5","SUS6","SUS7","SUS8","SUS9","SUS10"],
    "NASA-TLX": ["TLX1","TLX2","TLX3","TLX4","TLX5","TLX6"],
    "User Testing": ["UT1","UT2","UT3","UT4","UT5","UT6","UT7"]
}

fw_embeddings = {fw: minilm.encode(descs, convert_to_tensor=True) for fw, descs in FRAMEWORKS.items()}

def chunk(text):
    words = text.split()
    if len(words) <= 400:
        return [text]
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start+400]))
        start += 350
    return chunks

def clean(text):
    doc = nlp(text)
    tokens = [t.lemma_.lower() for t in doc if not t.is_stop and not t.is_punct and not t.is_space]
    urgency = [t for t in tokens if t in {"struggle","confuse","fail","frustrate","break","stuck","error","crash"}]
    return " ".join(tokens), urgency

def get_framework(text):
    res = classifier(text, list(FRAMEWORKS.keys()))
    label, conf = res["labels"][0], res["scores"][0]
    return (label, round(conf, 3)) if conf >= 0.4 else ("unclassified", round(conf, 3))

def get_subconcept(text, fw):
    if fw == "unclassified":
        return None, None
    emb = minilm.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(emb, fw_embeddings[fw])[0]
    idx = int(scores.argmax())
    labels = FRAMEWORK_LABELS[fw]
    return labels[idx] if idx < len(labels) else None, FRAMEWORKS[fw][idx]

def get_sentiment_and_attention(text):
    inputs = rb_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = rb_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    label = rb_model.config.id2label[int(probs.argmax())].lower()
    score = round(float(probs.max()), 3)
    attn = outputs.attentions[-1].mean(dim=1).mean(dim=1)[0]
    tokens = rb_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    weights = sorted(
        [{"word": t.replace("Ġ","").replace("##",""), "weight": round(float(w),3)}
         for t, w in zip(tokens, attn.tolist())
         if t not in rb_tokenizer.all_special_tokens and t.replace("Ġ","").replace("##","")],
        key=lambda x: x["weight"], reverse=True
    )[:8]
    return label, score, weights

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    body = request.get_json(silent=True)
    if not body or not body.get("text","").strip():
        return jsonify({"error": "missing text"}), 400

    text = body["text"].strip()

    try:
        if lang_detect(text) != "en":
            return jsonify({"error": "only English supported"}), 400
    except:
        pass

    results = []
    for c in chunk(text):
        cleaned, urgency = clean(c)
        if len(cleaned.split()) < 10:
            continue
        fw, conf = get_framework(cleaned)
        sub_id, sub_label = get_subconcept(cleaned, fw)
        sentiment, sent_score, attention = get_sentiment_and_attention(c)
        results.append({
            "framework": fw, "confidence": conf,
            "sub_id": sub_id, "sub_label": sub_label,
            "sentiment": sentiment, "sentiment_score": sent_score,
            "attention": attention, "urgency": urgency
        })

    if not results:
        return jsonify({"error": "text too short"}), 422

    from collections import Counter
    primary = Counter(r["framework"] for r in results).most_common(1)[0][0]
    pc = [r for r in results if r["framework"] == primary]
    avg_sent = round(sum(r["sentiment_score"] for r in pc) / len(pc), 3)
    all_urgency = [u for r in pc for u in r["urgency"]]
    crit = 1.0 if any(k in all_urgency for k in ["checkout","login","payment"]) else 0.5 if any(k in all_urgency for k in ["navigate","search"]) else 0.3
    severity = min(4, round(((len(pc)/len(results))*0.4 + avg_sent*0.4 + crit*0.2) * 4))

    top_words = {}
    for r in pc:
        for w in r["attention"]:
            top_words[w["word"]] = max(top_words.get(w["word"], 0), w["weight"])

    return jsonify({
        "framework": primary,
        "sub_concept": {"id": pc[0]["sub_id"], "label": pc[0]["sub_label"]},
        "sentiment": {"label": pc[0]["sentiment"], "score": avg_sent},
        "severity": {"score": severity, "label": ["Negligible","Low","Medium","High","Critical"][severity]},
        "top_attention_words": sorted([{"word":w,"weight":v} for w,v in top_words.items()], key=lambda x: x["weight"], reverse=True)[:8],
        "chunks_analyzed": len(results)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
