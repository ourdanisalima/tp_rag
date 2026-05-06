import faiss
import numpy as np
import json
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
 
FICHIER_XLSX = "/home/amilas/tp_rag/CIS_RCP_export.xlsx"
DATA_DIR = "data/"
 
COLONNES_TEXTE = [
    "denomination",
    "forme_pharmaceutique",
    "composition",
    "indications",
    "posologie",
    "contre_indications",
    "mises_en_garde",
    "interactions",
    "grossesse_allaitement",
    "effets_indesirables",
    "surdosage",
    "conditions_prescription",
]
 
 
def nettoyer(valeur) -> str:
    if pd.isna(valeur) or valeur is None:
        return ""
    return str(valeur).strip()
 
 
def medicament_to_text(row: pd.Series) -> str:
    parties = []
    labels = {
        "denomination": "Médicament",
        "forme_pharmaceutique": "Forme pharmaceutique",
        "composition": "Composition",
        "indications": "Indications thérapeutiques",
        "posologie": "Posologie",
        "contre_indications": "Contre-indications",
        "mises_en_garde": "Mises en garde",
        "interactions": "Interactions médicamenteuses",
        "grossesse_allaitement": "Grossesse et allaitement",
        "effets_indesirables": "Effets indésirables",
        "surdosage": "Surdosage",
        "conditions_prescription": "Conditions de prescription",
    }
    for col, label in labels.items():
        val = nettoyer(row.get(col, ""))
        if val:
            parties.append(f"{label} : {val}")
    return "\n".join(parties)
 
 
def chunker(texte: str, taille_max: int = 600, overlap: int = 80) -> list:
    if len(texte) <= taille_max:
        return [texte]
 
    separateurs = ["\n\n", "\n", ". ", " "]
 
    def decouper(t, sep_idx=0):
        if len(t) <= taille_max:
            return [t]
        sep = separateurs[sep_idx] if sep_idx < len(separateurs) else " "
        parties = t.split(sep)
        result = []
        courant = ""
        for partie in parties:
            if len(courant) + len(partie) + len(sep) <= taille_max:
                courant += (sep if courant else "") + partie
            else:
                if courant:
                    result.append(courant)
                if len(partie) > taille_max and sep_idx + 1 < len(separateurs):
                    result.extend(decouper(partie, sep_idx + 1))
                else:
                    courant = partie
        if courant:
            result.append(courant)
        return result
 
    return decouper(texte)
 
 
def construire_corpus(df: pd.DataFrame) -> list:
    documents = []
    for _, row in df.iterrows():
        texte = medicament_to_text(row)
        if len(texte.strip()) < 30:
            continue
 
        denomination = nettoyer(row.get("denomination", "inconnu"))
        code_cis = str(nettoyer(row.get("code_cis", "")))
 
        chunks = chunker(texte)
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                documents.append({
                    "id": f"{code_cis}_chunk_{i}",
                    "contenu": chunk,
                    "metadata": {
                        "medicament": denomination,
                        "cis": code_cis,
                        "numero_amm": nettoyer(row.get("numero_amm", "")),
                        "titulaire": nettoyer(row.get("titulaire_amm", "")),
                        "date_maj": nettoyer(row.get("date_mise_a_jour", "")),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    }
                })
    return documents
 
 
def embedder_chunks(documents: list, modele) -> np.ndarray:
    textes = [d["contenu"] for d in documents]
    print(f"  Encodage de {len(textes)} chunks...")
    vecteurs = modele.encode(textes, show_progress_bar=True, batch_size=32)
    return np.array(vecteurs, dtype=np.float32)
 
 
def creer_index_faiss(vecteurs: np.ndarray) -> faiss.Index:
    dim = vecteurs.shape[1]
    faiss.normalize_L2(vecteurs)
    index = faiss.IndexFlatIP(dim)
    index.add(vecteurs)
    return index
 
 
def sauvegarder_index(index, documents: list, chemin: str = DATA_DIR):
    os.makedirs(chemin, exist_ok=True)
    faiss.write_index(index, os.path.join(chemin, "index.faiss"))
    with open(os.path.join(chemin, "documents.json"), "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    print(f"  Index sauvegardé : {index.ntotal} vecteurs")
    print(f"  Documents sauvegardés : {len(documents)} chunks")
 
 
def main():
    print("=== PHASE 1 : INDEXATION ===\n")
 
    print(f"[1/4] Chargement du fichier : {FICHIER_XLSX}")
    df = pd.read_excel(FICHIER_XLSX)
    print(f"  {len(df)} médicaments chargés, {len(df.columns)} colonnes")
 
    print("[2/4] Construction des chunks...")
    documents = construire_corpus(df)
    print(f"  {len(documents)} chunks issus de {len(df)} médicaments")
 
    if not documents:
        print("[!] Aucun document généré. Vérifiez le fichier Excel.")
        return
 
    print("[3/4] Chargement du modèle d'embedding...")
    modele = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
 
    print("[4/4] Création des embeddings et de l'index FAISS...")
    vecteurs = embedder_chunks(documents, modele)
    print(f"  Dimension des vecteurs : {vecteurs.shape[1]}")
 
    index = creer_index_faiss(vecteurs)
    sauvegarder_index(index, documents)
 
    print("\n=== INDEXATION TERMINÉE ===")
    print("Lancez maintenant : python rag.py")
 
 
if __name__ == "__main__":
    main()
 
