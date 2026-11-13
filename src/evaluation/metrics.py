"""Metriche di retrieval, riusabili anche nella Fase 6.

sim[i, j]: similarita' tra la query i e il candidato j.
relevant[i, j]: True se il candidato j e' una risposta giusta per la query i.
Per ogni query conta il rango del primo candidato giusto (1 = in cima alla lista).
"""
import numpy as np


def exact_relevance(n):
    """Matrice di rilevanza esatta: solo il candidato con lo stesso indice e' giusto."""
    # giusto solo il candidato con lo stesso narration_id (query i <-> candidato i)
    return np.eye(n, dtype=bool)


def class_relevance(verb, noun):
    """Matrice di rilevanza per classe: giusti tutti i candidati con verbo e nome uguali."""
    # giusto ogni candidato con lo stesso verbo e lo stesso nome della query
    return (verb[:, None] == verb[None, :]) & (noun[:, None] == noun[None, :])


def first_relevant_rank(sim, relevant):
    """Rango (da 1) del primo candidato giusto per ogni query."""
    if not relevant.any(1).all():
        raise ValueError("ogni query deve avere almeno un candidato giusto")
    # niente ordinamento (argsort su n x n e' lento e occupa molta memoria con val grandi):
    # il primo giusto e' quello con la similarita' piu' alta tra i giusti, e il suo rango
    # e' 1 + quanti candidati hanno una similarita' strettamente maggiore della sua
    best_relevant = np.where(relevant, sim, -np.inf).max(1)
    return (sim > best_relevant[:, None]).sum(1) + 1


def retrieval_metrics(sim, relevant, ks=(1, 5, 10)):
    """Calcola R@k per ogni k in ks e il rango mediano."""
    ranks = first_relevant_rank(sim, relevant)
    metrics = {f"R@{k}": float((ranks <= k).mean() * 100) for k in ks}
    metrics["rango_mediano"] = float(np.median(ranks))
    return metrics
