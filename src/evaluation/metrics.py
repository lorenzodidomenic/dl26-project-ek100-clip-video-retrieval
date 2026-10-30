"""Metriche di retrieval, riusabili anche nella Fase 6.

sim[i, j]: similarita' tra la query i e il candidato j.
relevant[i, j]: True se il candidato j e' una risposta giusta per la query i.
Per ogni query conta il rango del primo candidato giusto (1 = in cima alla lista).
"""
import numpy as np


def exact_relevance(n):
    # giusto solo il candidato con lo stesso narration_id (query i <-> candidato i)
    return np.eye(n, dtype=bool)


def class_relevance(verb, noun):
    # giusto ogni candidato con lo stesso verbo e lo stesso nome della query
    return (verb[:, None] == verb[None, :]) & (noun[:, None] == noun[None, :])


def first_relevant_rank(sim, relevant):
    if not relevant.any(1).all():
        raise ValueError("ogni query deve avere almeno un candidato giusto")
    order = np.argsort(-sim, axis=1)                        # candidati dal piu' simile al meno simile
    hits = np.take_along_axis(relevant, order, axis=1)      # True dove, in quell'ordine, c'e' un giusto
    return hits.argmax(1) + 1                               # posizione del primo giusto, da 1


def retrieval_metrics(sim, relevant, ks=(1, 5, 10)):
    ranks = first_relevant_rank(sim, relevant)
    metrics = {f"R@{k}": float((ranks <= k).mean() * 100) for k in ks}
    metrics["rango_mediano"] = float(np.median(ranks))
    return metrics
