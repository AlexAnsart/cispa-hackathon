# Commandes Rapides - Phase 1 (UPDATED)

## 🚀 Trois modes d'exécution disponibles

### Mode RAPIDE (Recommandé pour débuter) - ~6-8 minutes
```bash
sbatch run_solver_FAST.sh
```
- **Durée**: 6-8 minutes pour 100 images
- **Config**: 3 BS steps, 4 restarts, 60 PGD steps
- **Qualité**: Légèrement inférieure mais compétitive
- **Usage**: Tests rapides, itérations multiples

### Mode ÉQUILIBRÉ (Défaut) - ~8-12 minutes
```bash
sbatch run_solver.sh
```
- **Durée**: 8-12 minutes pour 100 images
- **Config**: 4 BS steps, 5 restarts, 80 PGD steps
- **Qualité**: Bon compromis vitesse/qualité
- **Usage**: Production standard

### Mode QUALITÉ (Pour submission finale) - ~60-90 minutes
```bash
sbatch run_solver_QUALITY.sh
```
- **Durée**: 60-90 minutes pour 100 images
- **Config**: 8 BS steps, 15 restarts, 150 PGD steps
- **Qualité**: Maximale
- **Usage**: Submission finale pour leaderboard

---

## 📊 Analyse des résultats - DEUX MODES

### Mode LOCAL (Rapide, SANS API) ⚡
```bash
python analyze.py output/submission_fast.npz --mode local
```

**Avantages**:
- ✅ Instantané (quelques secondes)
- ✅ Pas de rate limit
- ✅ Pas de consommation d'API
- ✅ Donne une **borne inférieure** du score

**Sortie**:
```
L2 Distance Statistics:
  Average (normalized): 0.1234  ← Meilleur cas possible
  Min:                  0.0543
  Max:                  0.9821

NOTE: Ceci est une BORNE INFÉRIEURE.
      Le score réel sera >= à cette valeur.
```

**Utilisation**:
- Tester rapidement plusieurs runs
- Comparer différentes configurations
- Vérifier la qualité avant de consumer l'API

### Mode API (Score réel, avec rate limit) 🌐
```bash
python analyze.py output/submission_fast.npz --mode api
```

⚠️ **Rate limit**: 15 minutes entre appels

**Sortie**:
```
Success Rate: 87/100 (87.0%)
Leaderboard Score: 0.1876  ← Score RÉEL
  Successful only:   0.1234
  Failed (all 1.0):  1.0000

Per-Image Results:
 ID | True | Pred | Status  | L2 Raw   | Score
-------------------------------------------------
  0 |   42 |   17 | SUCCESS |   4.2314 | 0.1460
  2 |   88 |   88 | FAILED  |   5.1234 | 1.0000
```

**Utilisation**:
- Obtenir le score réel avant soumission
- Identifier quelles images échouent
- Calibrer les kappas (Phase 2)

---

## 🎯 Workflow recommandé

### Itération rapide (développement)
```bash
# 1. Run rapide (6-8 min)
sbatch run_solver_FAST.sh

# 2. Attendre fin du job
tail -f logs/slurm_*.out

# 3. Analyse locale (instantanée, pas d'API)
python analyze.py output/submission_fast.npz --mode local

# 4. Si L2 moyen < 0.18, tester avec API
python analyze.py output/submission_fast.npz --mode api

# 5. Ajuster paramètres et recommencer
```

### Submission finale (compétition)
```bash
# 1. Run qualité maximale (60-90 min)
sbatch run_solver_QUALITY.sh

# 2. Analyse complète avec API
python analyze.py output/submission_quality.npz --mode api

# 3. Si success rate > 85%, soumettre
python submit.py output/submission_quality.npz
```

---

## 📈 Comparaison des modes

| Mode | Durée | BS steps | Restarts | PGD steps | Score attendu |
|------|-------|----------|----------|-----------|---------------|
| FAST | 6-8 min | 3 | 4 | 60 | 0.18-0.25 |
| ÉQUILIBRÉ | 8-12 min | 4 | 5 | 80 | 0.16-0.22 |
| QUALITÉ | 60-90 min | 8 | 15 | 150 | 0.15-0.20 |

**Stratégie intelligente**:
1. Débuter avec FAST pour tester
2. Itérer rapidement avec analyse locale
3. Quand satisfait, lancer QUALITÉ pour submission finale

---

## 🔍 Quand utiliser quel mode d'analyse ?

### Analyse LOCAL (--mode local)
**Utiliser quand**:
- ✅ Tu veux tester plusieurs runs rapidement
- ✅ Tu développes/debugges
- ✅ Tu veux éviter de consumer les rate limits API
- ✅ Tu veux une estimation rapide "best case"

**Ne pas utiliser si**:
- ❌ Tu veux le score réel (API obligatoire)
- ❌ Tu veux identifier quelles images échouent

### Analyse API (--mode api)
**Utiliser quand**:
- ✅ Tu veux le score RÉEL avant soumission
- ✅ Tu veux identifier images qui échouent (pour Phase 2)
- ✅ Tu es prêt à attendre 15 min avant prochain appel

**Ne pas utiliser si**:
- ❌ Tu itères rapidement (use local)
- ❌ Tu as déjà appelé l'API il y a < 15 min

---

## 📊 Surveillance en temps réel

```bash
# Dashboard complet (relancer pour rafraîchir)
python monitor.py

# Logs en temps réel
tail -f logs/slurm_*.out

# Statut SLURM
squeue -u $USER

# Stats rapides JSON
cat logs/stats_summary.json

# Dernières lignes du log
tail -n 20 logs/slurm_*.out
```

---

## 📤 Soumission

```bash
# Soumettre (rate limit: 5 min)
python submit.py output/submission_fast.npz

# Logits + Submit en une fois
python submit.py output/submission_fast.npz --action both
```

---

## 🎯 Exemples concrets

### Exemple 1: Premier test rapide
```bash
# 1. Lancer FAST mode
sbatch run_solver_FAST.sh

# 2. Attendre 6-8 min, puis analyser localement
python analyze.py output/submission_fast.npz --mode local

# Résultat: "Average (normalized): 0.1834"
# → Bon signe, on teste avec API

# 3. Vérifier score réel
python analyze.py output/submission_fast.npz --mode api

# Résultat: "Success Rate: 82%, Score: 0.2145"
# → OK mais peut mieux faire
```

### Exemple 2: Comparer plusieurs configs
```bash
# Run 1: FAST
sbatch run_solver_FAST.sh
# → Attendre 6-8 min
python analyze.py output/submission_fast.npz --mode local
# → Score local: 0.1834

# Run 2: ÉQUILIBRÉ (modifier epsilon range dans run_solver.sh)
sbatch run_solver.sh
# → Attendre 8-12 min
python analyze.py output/submission_run2.npz --mode local
# → Score local: 0.1612

# Run 2 est meilleur ! On le teste avec API
python analyze.py output/submission_run2.npz --mode api
# → Score réel: 0.1876, Success: 88%

# Bon score, on soumet celui-là
python submit.py output/submission_run2.npz
```

### Exemple 3: Pipeline complet
```bash
# Phase 1: Test rapide
sbatch run_solver_FAST.sh
python monitor.py  # Surveiller
python analyze.py output/submission_fast.npz --mode local
# → 0.18, bon

# Phase 2: Qualité pour finale
sbatch run_solver_QUALITY.sh
# → Café pendant 60-90 min ☕
python analyze.py output/submission_quality.npz --mode api
# → Success: 89%, Score: 0.1734

# Phase 3: Submit
python submit.py output/submission_quality.npz
# → Leaderboard updated!
```

---

## 🐛 Dépannage

### "Rate limit exceeded" sur API
```bash
# Attendre 15 min OU utiliser mode local:
python analyze.py output/submission.npz --mode local
```

### Job trop lent
```bash
# Utiliser FAST mode au lieu de run_solver.sh:
sbatch run_solver_FAST.sh
```

### Score local vs API très différent
```
Local (borne inf): 0.1234
API (réel):        0.2876
```
**Cause**: Beaucoup d'images échouent (success rate faible)

**Solution**: Augmenter kappas ou utiliser QUALITY mode

---

## 📁 Fichiers générés

### Par mode FAST
- `output/submission_fast.npz`

### Par mode ÉQUILIBRÉ (run_solver.sh)
- `output/submission_run<N>.npz`

### Par mode QUALITY
- `output/submission_quality.npz`

### Logs communs
- `logs/local_state.json` (état persistant)
- `logs/run_history.json` (historique)
- `logs/stats_summary.json` (stats)

---

## ⚡ Résumé ultra-rapide

```bash
# Test rapide (6-8 min)
sbatch run_solver_FAST.sh
python analyze.py output/submission_fast.npz --mode local

# Production (8-12 min)
sbatch run_solver.sh
python analyze.py output/submission_run1.npz --mode api

# Qualité max (60-90 min)
sbatch run_solver_QUALITY.sh
python analyze.py output/submission_quality.npz --mode api
python submit.py output/submission_quality.npz
```

---

## 🎓 Pourquoi deux modes d'analyse ?

**Mode local**: Calcule seulement les distances L2 entre images originales et adverses
- Assume que toutes les attaques réussissent (optimiste)
- Borne inférieure du score (ne peut être que pire en réalité)

**Mode API**: Obtient les prédictions du black-box
- Voit quelles images sont réellement misclassified
- Score réel: succès = L2, échec = 1.0
- C'est ce score qui compte pour le leaderboard

**En pratique**:
- Use local pour itérer vite (10+ fois par heure)
- Use API pour valider (max 4 fois par heure à cause du rate limit)

---

**Maintenant tu as trois vitesses et deux modes d'analyse. Utilise intelligemment ! 🚀**

