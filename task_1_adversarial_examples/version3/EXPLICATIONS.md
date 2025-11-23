# Explications: Epsilon, Margin, et L2 Normalisé

## 📊 L2 Distance (Normalisée vs Brute)

### L2 Brute (Raw)
La distance L2 **brute** est la distance euclidienne dans l'espace pixel:
```
L2_raw = sqrt(sum((x_adv - x_orig)²))
```

Pour une image 28×28×3:
- Chaque pixel peut varier de 0 à 1 (valeurs normalisées)
- Maximum théorique: sqrt(3 × 28 × 28) = sqrt(2352) ≈ **48.5**
- Valeurs typiques: **2-8** pour des attaques réussies

**Exemple**: L2 brute = 2.125 signifie que la perturbation totale est de 2.125 unités dans l'espace pixel.

### L2 Normalisée (Normalized)
La distance L2 **normalisée** est divisée par le maximum théorique:
```
L2_norm = L2_raw / sqrt(C × H × W)
       = L2_raw / sqrt(3 × 28 × 28)
       = L2_raw / 48.5
```

**Propriétés**:
- Toujours entre **0 et 1**
- **0.0** = image identique (pas de perturbation)
- **1.0** = perturbation maximale théorique
- Valeurs typiques: **0.04-0.18** pour des attaques réussies

**Exemple**: 
- L2 brute = 2.125
- L2 normalisée = 2.125 / 48.5 ≈ **0.044** (excellent !)

### Pourquoi les deux ?
- **L2 brute**: Utile pour comprendre la magnitude réelle de la perturbation
- **L2 normalisée**: C'est ce que le **leaderboard utilise** pour le score
  - Score = L2 normalisée si attaque réussie
  - Score = 1.0 si attaque échouée (pénalité maximale)

---

## ε (Epsilon) - Taille Maximale de Perturbation

### Définition
**Epsilon (ε)** est la **taille maximale** de la perturbation autorisée, mesurée en **norme L2 brute**.

```
||x_adv - x_orig||_2 ≤ ε
```

### Dans notre algorithme
- **Recherche binaire** sur ε pour trouver le **minimum** qui réussit
- Plage de recherche: `[epsilon_min, epsilon_max]`
  - Mode FAST: `[1.5, 8.0]`
  - Mode QUALITY: `[0.5, 12.0]`

### Exemple
Si ε = 6.5:
- La perturbation peut être **au maximum** de 6.5 unités L2 brute
- L'algorithme cherche la **plus petite** perturbation ≤ 6.5 qui réussit
- Si trouvé à L2 = 4.2, alors ε_used = 4.2 (ou moins)

### Interprétation
- **ε faible** (2-4): Perturbation subtile, difficile à trouver
- **ε moyen** (4-8): Perturbation modérée, bon compromis
- **ε élevé** (8-12): Perturbation visible, facile à trouver

**Objectif**: Trouver le **minimum ε** qui réussit (minimiser L2).

---

## Margin (Marge de Confiance)

### Définition
**Margin** est la différence entre le logit de la classe prédite (fausse) et le logit de la vraie classe:

```
Margin = logit_max_wrong - logit_true
```

Où:
- `logit_max_wrong`: Le logit le plus élevé parmi les **mauvaises** classes
- `logit_true`: Le logit de la **vraie** classe

### Interprétation

**Margin > 0** (positif):
- ✅ **Attaque réussie** !
- Le modèle prédit une classe **fausse** avec plus de confiance que la vraie
- Plus le margin est élevé, plus l'attaque est "confiante"

**Margin < 0** (négatif):
- ❌ **Attaque échouée**
- Le modèle prédit encore la **vraie** classe
- Il faut pousser plus fort (augmenter ε ou κ)

**Margin ≈ 0**:
- ⚠️ **Frontière de décision**
- Le modèle hésite entre vraie et fausse classe
- Risque de ne pas transférer au black-box

### Exemple Concret

Supposons 10 classes (0-9), vraie classe = 3:

**Avant attaque**:
```
Logits: [0.1, 0.2, 0.1, 5.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
         classe 0  1   2   3✓  4   5   6   7   8   9
```
- Classe prédite: 3 (correcte)
- logit_true = 5.2
- logit_max_wrong = max([0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) = 0.2
- Margin = 0.2 - 5.2 = **-5.0** (échec)

**Après attaque réussie**:
```
Logits: [0.1, 0.2, 0.1, 2.1, 0.1, 0.1, 0.1, 0.1, 0.1, 4.8]
         classe 0  1   2   3   4   5   6   7   8   9✓
```
- Classe prédite: 9 (fausse !)
- logit_true = 2.1
- logit_max_wrong = 4.8
- Margin = 4.8 - 2.1 = **+2.7** (succès !)

### Relation avec κ (Kappa)

**Kappa (κ)** est une **marge de confiance minimale** requise:

```
Succès si: Margin > κ
```

- **κ = 0.0**: Accepte n'importe quel margin positif (minimal L2)
- **κ = 5.0**: Exige un margin de +5.0 minimum (plus sûr, L2 plus élevé)

**Stratégie**:
- Phase 1: κ = 0.0 (minimiser L2)
- Phase 2: Ajuster κ par image selon transfert API
  - Échec API → Augmenter κ (pousser plus fort)
  - Succès API avec margin énorme → Diminuer κ (réduire L2)

---

## 📈 Exemple Complet d'Affichage

```
[  1/100] Image ID   0 (Label:   0, κ=0.00)
  ✓ SUCCESS
  L2: 2.1250 (norm: 0.0438) | ε: 2.125 | Margin: +4.65 | Time: 12.3s
  BS steps with success: 2/2
  [ε = max L2 perturbation allowed (raw), Margin = logit_wrong - logit_true]
```

**Interprétation**:
- **L2: 2.1250 (norm: 0.0438)**: 
  - Perturbation brute de 2.125 unités
  - Normalisée à 0.0438 (excellent, très faible !)
- **ε: 2.125**: 
  - L'algorithme a trouvé une perturbation de 2.125 unités
  - C'est le minimum trouvé qui réussit
- **Margin: +4.65**: 
  - Le modèle prédit la mauvaise classe avec +4.65 logits de plus que la vraie
  - Attaque très confiante (bon signe pour le transfert)
- **BS steps with success: 2/2**: 
  - Les 2 étapes de recherche binaire ont trouvé des succès
  - L'algorithme a convergé rapidement

---

## 🎯 Objectifs et Métriques

### Score Leaderboard
```
Score = moyenne de tous les scores par image
```

Score par image:
- **Si attaque réussie**: Score = L2 normalisée (0.0 à 1.0)
- **Si attaque échouée**: Score = 1.0 (pénalité maximale)

**Objectif**: Minimiser le score (parfait = 0.0, pire = 1.0)

### Métriques Clés

| Métrique | Objectif | Interprétation |
|----------|----------|----------------|
| **L2 normalisée** | < 0.15 | Perturbation faible (invisible) |
| **Success rate** | > 85% | La plupart des attaques réussissent |
| **Margin moyen** | > +2.0 | Attaques confiantes (bon transfert) |
| **Score leaderboard** | < 0.20 | Compétitif |

### Trade-offs

**L2 vs Margin**:
- L2 faible + Margin faible → Risque d'échec au transfert
- L2 élevé + Margin élevé → Sûr mais perturbation visible
- **Objectif**: L2 faible + Margin suffisant (> κ)

**Epsilon vs Temps**:
- Epsilon élevé → Trouve plus facilement mais L2 plus élevé
- Epsilon faible → L2 plus faible mais recherche plus longue
- **Stratégie**: Recherche binaire pour trouver le minimum

---

## 🔧 Comment Ajuster

### Si L2 trop élevé (> 0.20 normalisé)
- Réduire `--epsilon-max` (ex: 8.0 → 6.0)
- Augmenter `--bs-steps` (ex: 2 → 4) pour recherche plus fine
- Réduire `--kappa` si > 0 (accepter margin plus faible)

### Si Success rate trop faible (< 80%)
- Augmenter `--epsilon-max` (ex: 8.0 → 12.0)
- Augmenter `--restarts` (ex: 2 → 4) pour plus de chances
- Augmenter `--kappa` (ex: 0.0 → 2.0) pour pousser plus fort

### Si Margin trop faible (< +1.0)
- Augmenter `--kappa` (ex: 0.0 → 3.0)
- Augmenter `--epsilon-max` pour permettre plus de perturbation

---

## 📚 Résumé Ultra-Rapide

- **L2 brute**: Distance réelle (typique: 2-8)
- **L2 normalisée**: Score utilisé (typique: 0.04-0.18)
- **ε (epsilon)**: Taille max perturbation autorisée (recherche binaire)
- **Margin**: Confiance de l'attaque (logit_wrong - logit_true)
- **κ (kappa)**: Margin minimum requis (ajustable par image)

**Objectif final**: Minimiser L2 normalisée tout en maintenant success rate > 85%.

