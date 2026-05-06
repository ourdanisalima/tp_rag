import faiss
import numpy as np
import json
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
 
load_dotenv()
 
DATA_DIR = "data/"
SEUIL_CONFIANCE = 0.3
 
 
def charger_index(chemin: str = DATA_DIR):
    index = faiss.read_index(os.path.join(chemin, "index.faiss"))
    with open(os.path.join(chemin, "documents.json"), "r", encoding="utf-8") as f:
        documents = json.load(f)
    return index, documents
 
 
def rechercher(question: str, modele, index, documents: list, k: int = 4) -> list:
    vecteur = modele.encode([question], dtype=np.float32)
    faiss.normalize_L2(vecteur)
    scores, indices = index.search(vecteur, k)
 
    resultats = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        resultats.append({
            "contenu": documents[idx]["contenu"],
            "metadata": documents[idx]["metadata"],
            "score": float(score),
        })
    return resultats
 
 
def construire_prompt_systeme() -> str:
    return """Tu es un assistant spécialisé en information sur les médicaments, basé sur les notices officielles françaises (BDPM - Base de Données Publique des Médicaments).
 
Règles absolues :
1. Tu réponds UNIQUEMENT à partir des informations fournies dans le contexte. Tu n'inventes jamais.
2. Si l'information n'est pas dans le contexte, tu le dis clairement : "Je ne trouve pas cette information dans ma base de données."
3. Tu indiques toujours le nom du médicament concerné pour chaque information citée.
4. Chaque réponse se termine obligatoirement par : "⚠️ Ces informations ne remplacent pas l'avis d'un professionnel de santé. En cas de doute, consultez votre médecin ou votre pharmacien."
5. Tu n'interprètes jamais, tu cites les informations officielles.
6. Si la question concerne un médicament hors de ta base, tu le signales explicitement.
 
Format de réponse : clair, structuré, avec le nom du médicament source visible."""
 
 
def generer_reponse(question: str, chunks_pertinents: list, historique: list, client: Groq) -> str:
    if not chunks_pertinents or chunks_pertinents[0]["score"] < SEUIL_CONFIANCE:
        return (
            "Je ne trouve pas d'information directement pertinente dans ma base de données pour cette question.\n\n"
            "⚠️ Ces informations ne remplacent pas l'avis d'un professionnel de santé. "
            "En cas de doute, consultez votre médecin ou votre pharmacien."
        )
 
    contexte_parts = []
    for i, chunk in enumerate(chunks_pertinents):
        med = chunk["metadata"]["medicament"]
        contexte_parts.append(f"[Source {i+1} — {med}]\n{chunk['contenu']}")
    contexte = "\n\n".join(contexte_parts)
 
    prompt_user = f"""Contexte issu des notices officielles :
{contexte}
 
Question : {question}"""
 
    messages = [{"role": "system", "content": construire_prompt_systeme()}]
    messages.extend(historique[-6:])
    messages.append({"role": "user", "content": prompt_user})
 
    reponse = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1000,
        temperature=0.2,
    )
 
    return reponse.choices[0].message.content
 
 
def afficher_sources(chunks: list):
    meds_vus = set()
    sources = []
    for chunk in chunks:
        med = chunk["metadata"]["medicament"]
        if med not in meds_vus:
            meds_vus.add(med)
            sources.append(f"  • {med} (score: {chunk['score']:.2f})")
    if sources:
        print("\nSources consultées :")
        print("\n".join(sources))
 
 
def main():
    print("Chargement de la base de connaissances...")
 
    if not os.path.exists(os.path.join(DATA_DIR, "index.faiss")):
        print("[!] Index non trouvé. Lancez d'abord : python indexation.py")
        return
 
    index, documents = charger_index()
    modele = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
 
    print(f"Base chargée : {index.ntotal} vecteurs, {len(documents)} chunks")
    print("Système RAG prêt. Tapez 'quit' pour quitter.\n")
 
    historique = []
 
    while True:
        try:
            question = input("Votre question : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break
 
        if question.lower() in ["quit", "exit", "q"]:
            print("Au revoir !")
            break
 
        if not question:
            continue
 
        chunks = rechercher(question, modele, index, documents, k=4)
        reponse = generer_reponse(question, chunks, historique, client)
 
        print(f"\nRéponse :\n{reponse}")
        afficher_sources(chunks)
        print()
 
        historique.append({"role": "user", "content": question})
        historique.append({"role": "assistant", "content": reponse})
 
 
if __name__ == "__main__":
    main()
 
