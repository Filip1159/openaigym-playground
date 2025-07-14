# 🧠 Notatka z nauki: Reinforcement Learning (RL) — podstawy

## 📦 Środowisko (*Environment*)

- To **świat**, w którym działa agent.
- Definiuje:
  - **przestrzeń stanów** (`observation space`) — np. pozycja i prędkość w CartPole
  - **przestrzeń akcji** (`action space`) — np. ruch w lewo/prawo
  - **nagrody** za wykonane akcje
- W `OpenAI Gym` tworzy się środowisko np. tak:
  ```python
  env = gym.make("CartPole-v1", render_mode=None)
  ```

## 👤 Agent

- **Uczy się** na podstawie obserwacji i nagród, aby maksymalizować sumę nagród w czasie.
- W każdej chwili wybiera akcję na podstawie aktualnego stanu.
- W RL agent często reprezentowany jest jako:
  - Funkcja decyzyjna (np. `policy`)
  - Model (np. sieć neuronowa, która aproksymuje wartość akcji)

## 🎭 Aktor i Krytyk (*Actor & Critic*)

- **Actor** – komponent, który **podejmuje decyzje** (akcje), często reprezentowany jako `π(s)`
- **Critic** – komponent, który **ocenia jakość działania aktora**, np. wyznaczając wartość stanu `V(s)` lub wartość akcji `Q(s, a)`
- W `DQN` (Deep Q-Network) krytyk to sieć estymująca Q-funkcję (patrz niżej), a aktorem jest algorytm oparty na `argmax(Q(s, a))`.

## 🟰 Q-funkcja (`Q(s, a)`)

- Szacuje **oczekiwaną sumę nagród**, jaką agent może uzyskać, wykonując akcję `a` w stanie `s` i postępując dalej optymalnie.
- Używana np. w `Q-learningu` i `DQN`.

Formuła aktualizacji (tablicowy Q-learning):

```
Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]
```

## 🔤 Ważne parametry

| Symbol | Nazwa        | Znaczenie |
|--------|--------------|-----------|
| `α`    | **Learning rate** (tempo uczenia) | Jak bardzo nowa wiedza nadpisuje starą |
| `γ`    | **Discount factor** (czynnik dyskonta) | Jak bardzo agent dba o przyszłe nagrody |
| `ε`    | **Epsilon** (w eksploracji) | Prawdopodobieństwo losowego działania (eksploracja) |
| `r`    | **Reward** (nagroda) | Liczba otrzymana od środowiska |
| `s`, `s'` | Stan obecny / następny | Opis sytuacji w środowisku |
| `a`, `a'` | Akcja | Czynność podjęta przez agenta |

## ♻️ Epsilon-Greedy (eksploracja vs eksploatacja)

- Strategia działania:
  - Z prawdopodobieństwem `ε` agent wybiera **losową akcję** (eksploracja)
  - Z prawdopodobieństwem `1 - ε` wybiera **najlepszą znaną akcję** (eksploatacja)
- Typowy schemat: `ε` zmniejsza się w czasie (`decay`), aby na początku agent eksperymentował, a potem się stabilizował.

## 🧠 Deep Q-Learning (DQN)

- Używa sieci neuronowej do aproksymacji funkcji `Q(s, a)`
- W treningu minimalizujemy tzw. **loss** między `Q(s, a)` a celem:

```python
loss = mse(Q(s, a), r + γ * max_a' Q(s', a'))
```

## 🧪 Replay Buffer (bufor doświadczeń)

- Przechowuje przeżyte przez agenta doświadczenia `(s, a, r, s', done)`
- Podczas uczenia losujemy **mini-batch** z bufora, co poprawia stabilność i efektywność nauki (zmniejsza korelacje czasowe)

## 💡 Praktyczne wskazówki

- Nie renderuj środowiska podczas treningu (`render_mode=None`) – przyspiesza działanie.
- Użyj `render_mode="human"` tylko do demo po treningu.
- Pracuj z `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`, by łatwo przełączać CPU ↔ GPU.
