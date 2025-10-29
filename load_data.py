import pandas as pd
import requests
import zipfile
import io
import os
from pathlib import Path


def load_and_prepare_data(url: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Télécharge, extrait et prépare le jeu de données Sentiment140.

    Parameters
    ----------
    url : str
        URL du zip "trainingandtestdata.zip".
    data_dir : str
        Répertoire local où stocker les fichiers.

    Returns
    -------
    pd.DataFrame
        DataFrame avec colonnes ['sentiment', 'text'] et labels {0,1}.
    """
    # Prépare les chemins
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "sentiment140.zip"
    csv_path = data_dir / "training.1600000.processed.noemoticon.csv"

    # Télécharger le zip si nécessaire (uniquement si le CSV n'est pas déjà présent)
    if not csv_path.exists():
        if not zip_path.exists():
            print("Téléchargement du jeu de données...")
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
            except requests.RequestException as e:
                raise RuntimeError(f"Échec du téléchargement : {e}") from e

            # Sauvegarde du zip sur disque (plus robuste que de garder en mémoire)
            zip_path.write_bytes(resp.content)
            print(f"Zip téléchargé: {zip_path}")

        # Extraire le zip
        print("Extraction des fichiers...")
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(data_dir)
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Zip corrompu: {zip_path}") from e
        print("Extraction terminée.")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Fichier CSV introuvable après extraction: {csv_path}"
        )

    # Définir les colonnes et charger les données
    cols = ["sentiment", "id", "date", "query", "user", "text"]
    df = pd.read_csv(
        csv_path,
        header=None,
        names=cols,
        encoding="latin-1",  # encodage spécifique au dataset
        # low_memory=False  # décommentez si vous avez un warning de dtype
    )

    # Ne garder que ce qui est utile
    df = df[["sentiment", "text"]].copy()

    # Mapper les sentiments : 0 -> 0 (négatif), 4 -> 1 (positif)
    df["sentiment"] = df["sentiment"].replace({4: 1}).astype(int)

    print("Préparation des données terminée.")
    return df


if __name__ == "__main__":
    dataset_url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"

    # Charger
    df = load_and_prepare_data(dataset_url)

    # Pour des raisons de performance pendant le TP, on travaille sur un échantillon
    # Enlevez .sample(...) pour utiliser le jeu de données complet
    data_df = df.sample(n=50000, random_state=42)

    # Sauvegarder l'échantillon pour les étapes suivantes
    output_path = Path("data") / "raw_tweets.csv"
    data_df.to_csv(output_path, index=False)
    print(f"Échantillon de données sauvegardé dans {output_path}")
