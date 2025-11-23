# 🚀 START HERE - Guide Ultra-Rapide

## Commande à lancer MAINTENANT (mode rapide)

```bash
cd /p/home/jusers/ansart1/jureca/code/task_1_adversarial_examples/version3
sbatch run_solver_FAST.sh
```

**Durée**: 6-8 minutes ⚡

---

## Trois modes disponibles

| Mode | Commande | Durée | Score attendu | Usage |
|------|----------|-------|---------------|-------|
| ⚡ **RAPIDE** | `sbatch run_solver_FAST.sh` | **6-8 min** | 0.18-0.25 | Tests/itérations |
| ⚖️ Équilibré | `sbatch run_solver.sh` | 8-12 min | 0.16-0.22 | Production |
| 🎯 Qualité | `sbatch run_solver_QUALITY.sh` | 60-90 min | 0.15-0.20 | Finale |

**Recommandation**: Commence par RAPIDE pour tester !

---

## Analyser les résultats (DEUX modes)

### 1. Mode LOCAL (instantané, sans API) 
```bash
python analyze.py output/submission_fast.npz --mode local
```
- ✅ Rapide (secondes)
- ✅ Pas de rate limit
- ✅ Borne inférieure du score
- Usage: Itération rapide

### 2. Mode API (score réel, 15 min cooldown)
```bash
python analyze.py output/submission_fast.npz --mode api
```
- ✅ Score RÉEL
- ✅ Voit quelles images échouent
- ⚠️ Rate limit: 15 min entre appels
- Usage: Validation finale

---

## Workflow simple

```bash
# 1. Lancer (6-8 min)
sbatch run_solver_FAST.sh

# 2. Surveiller
tail -f logs/slurm_*.out

# 3. Analyser (instantané)
python analyze.py output/submission_fast.npz --mode local
# → Si L2 moyen < 0.20, c'est bon !

# 4. Vérifier score réel (15 min cooldown)
python analyze.py output/submission_fast.npz --mode api
# → Si success rate > 80%, excellent !

# 5. Soumettre (5 min cooldown)
python submit.py output/submission_fast.npz
```

---

## Interprétation rapide

### Analyse locale
```
Average (normalized): 0.1834
```
→ **Borne inférieure**. Score réel sera >= 0.1834

### Analyse API
```
Success Rate: 87/100 (87.0%)
Leaderboard Score: 0.1876
```
→ **Score réel**. C'est ce qui compte.

**Objectif**: Success > 85%, Score < 0.20

---

## Que faire maintenant ?

### Option 1: Test ultra-rapide (RECOMMANDÉ)
```bash
sbatch run_solver_FAST.sh
# → Attends 6-8 min
python analyze.py output/submission_fast.npz --mode local
# → Vois si c'est prometteur
```

### Option 2: Vérifier environnement d'abord
```bash
python preflight_check.py
# → Vérifie que tout est OK
sbatch run_solver_FAST.sh
```

### Option 3: Directement production
```bash
sbatch run_solver.sh
# → 8-12 min, meilleure qualité
```

---

## Fichiers importants

- **`COMMANDES_UPDATED.md`** ← Guide complet des commandes
- **`START_HERE.md`** ← Ce fichier (démarrage rapide)
- **`README.md`** ← Documentation technique

---

## Surveiller la progression

```bash
# Dashboard
python monitor.py

# Logs en direct
tail -f logs/slurm_*.out

# Job status
squeue -u $USER
```

---

## Questions fréquentes

**Q: Quel mode choisir ?**
A: FAST pour tester, QUALITY pour submission finale

**Q: Mode local ou API pour analyser ?**
A: Local pour itérer vite, API pour score réel (15 min cooldown)

**Q: Combien de temps ça prend ?**
A: FAST = 6-8 min, Standard = 8-12 min, QUALITY = 60-90 min

**Q: Quel score viser ?**
A: < 0.20 est compétitif, < 0.15 est excellent

**Q: Success rate minimum ?**
A: Viser > 85% (chaque échec coûte 1.0 au score)

---

## LANCE MAINTENANT ⚡

```bash
cd /p/home/jusers/ansart1/jureca/code/task_1_adversarial_examples/version3
sbatch run_solver_FAST.sh
```

**Résultats dans 6-8 minutes ! 🚀**

