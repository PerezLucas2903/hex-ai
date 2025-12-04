#!/usr/bin/env python3
"""
Jogar Hex contra um adversário (modelo de rede neural OU adversário aleatório).

Você é sempre o Jogador 1 (peças vermelhas, conectando topo–base).
O adversário é o Jogador 2 (peças azuis, conectando esquerda–direita).

Uso típico (na raiz do repo):

    # Contra adversário aleatório (equivalente ao usado no treino inicial)
    python play_human_vs_model.py --adversary-type random

    # Contra modelo de rede neural (Attention_QNet) com pesos específicos
    python play_human_vs_model.py --adversary-type nn --model-path models/attention_hex_adv.pth

    # Se não passar --model-path com adversary-type nn:
    # tenta automaticamente models/attention_hex_adv.pth e depois models/attention_hex.pth
"""

import argparse
import os
import sys

import numpy as np
import torch

from src.game.env import HEX
from src.game.adversary import NNAdversary, RandomAdversary
from src.models.attention import Attention_QNet


# ---------------------------------------------------------------------
# Argumentos de linha de comando
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play Hex against a trained model or a random adversary."
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=5,
        help="Tamanho do tabuleiro (n x n). Deve bater com o usado no treino (default: 5).",
    )
    parser.add_argument(
        "--adversary-type",
        type=str,
        choices=["nn", "random"],
        default="nn",
        help=(
            "Tipo de adversário: "
            "'nn' = rede neural (Attention_QNet); "
            "'random' = adversário aleatório (RandomAdversary). "
            "Default: nn."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Caminho para o checkpoint (.pth) da QNet usada pelo adversário NN. "
            "Se não for fornecido (e adversary-type=nn), tenta "
            "models/attention_hex_adv.pth e depois models/attention_hex.pth."
        ),
    )
    parser.add_argument(
        "--n-attention-layers",
        type=int,
        default=6,
        help="Número de camadas de atenção na Attention_QNet (default: 6).",
    )
    parser.add_argument(
        "--n-dim",
        type=int,
        default=32,
        help="Dimensão dos embeddings na Attention_QNet (default: 32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Dispositivo PyTorch (auto, cpu ou cuda). Default: auto.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        choices=["plot", "matrix"],
        default="plot",
        help="Modo de renderização do tabuleiro (plot ou matrix). Default: plot.",
    )
    parser.add_argument(
        "--n-best-actions",
        type=int,
        default=2,
        help=(
            "Para NNAdversary: número de melhores ações entre as quais ele escolhe "
            "aleatoriamente (default: 2). Ignorado em adversary-type=random."
        ),
    )
    return parser.parse_args()


def choose_device(arg_device: str) -> torch.device:
    if arg_device == "cpu":
        return torch.device("cpu")
    if arg_device == "cuda":
        if not torch.cuda.is_available():
            print("Aviso: CUDA não disponível, caindo para CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_model_path(args: argparse.Namespace) -> str | None:
    """Escolhe o caminho do modelo a carregar quando adversary-type = nn."""
    if args.model_path is not None:
        return args.model_path

    candidates = [
        os.path.join("models", "attention_hex_adv.pth"),
        os.path.join("models", "attention_hex.pth"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------
# Construção/carregamento de modelo para adversário NN
# ---------------------------------------------------------------------

def build_model(args: argparse.Namespace, device: torch.device) -> Attention_QNet:
    model = Attention_QNet(
        n_attention_layers=args.n_attention_layers,
        n_dim=args.n_dim,
    ).to(device)
    return model


def load_model_weights(model: torch.nn.Module, path: str, device: torch.device) -> None:
    """Carrega pesos do checkpoint.

    Tenta pegar a chave 'q_state'; se não existir, assume que o checkpoint é
    diretamente o state_dict.
    """
    print(f"Carregando modelo de: {path}")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "q_state" in checkpoint:
        state_dict = checkpoint["q_state"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print("Pesos carregados com sucesso.\n")


# ---------------------------------------------------------------------
# Interação com o humano
# ---------------------------------------------------------------------

def print_valid_actions_grid(env: HEX) -> None:
    """Imprime uma grade com os índices das casas livres,
    alinhada visualmente com o tabuleiro (0 fica embaixo à esquerda).
    """
    n = env.grid_size
    valid = set(env.get_valid_actions())
    print("\nÍndices das casas livres (alinhados ao tabuleiro):")

    # percorre as linhas de cima (n-1) para baixo (0)
    for r_display in range(n - 1, -1, -1):
        row_str = []
        for c in range(n):
            idx = r_display * n + c
            if idx in valid:
                row_str.append(f"{idx:2d}")
            else:
                row_str.append(" .")
        print(" ".join(row_str))

    print(
        f"\nDigite sua jogada como 'linha coluna' (1–{n}) "
        f"ou como um índice único (0–{n*n-1}). "
        f"Linha 1 corresponde à fileira de baixo do tabuleiro. 'q' para sair."
    )

def ask_human_action(env: HEX) -> int | None:
    """Lê uma ação do usuário. Retorna None se o usuário quiser sair."""
    n = env.grid_size
    valid_actions = set(env.get_valid_actions())

    while True:
        user_input = input("Sua jogada: ").strip()
        if user_input.lower() in {"q", "quit", "exit"}:
            return None

        parts = user_input.split()
        try:
            if len(parts) == 1:
                # formato: índice único
                idx = int(parts[0])
                if not (0 <= idx < n * n):
                    print(f"Índice fora do intervalo [0, {n*n - 1}]. Tente novamente.")
                    continue
            elif len(parts) == 2:
                # formato: linha coluna (1-based)
                row = int(parts[0]) - 1
                col = int(parts[1]) - 1
                if not (0 <= row < n and 0 <= col < n):
                    print(f"Coordenadas devem estar em 1–{n}. Tente novamente.")
                    continue
                idx = row * n + col
            else:
                print("Entrada inválida. Use 'linha coluna' ou 'índice'.")
                continue
        except ValueError:
            print("Entrada inválida. Use números inteiros ou 'q' para sair.")
            continue

        if idx not in valid_actions:
            print("Casa ocupada ou inválida. Escolha uma casa livre (veja a grade acima).")
            continue

        return idx


def force_empty_start(env: HEX):
    """Força o tabuleiro a começar vazio com o humano (Player 1) na vez.

    O reset padrão do ambiente às vezes deixa o adversário começar; aqui
    sobrescrevemos isso para garantir que o humano sempre comece.
    """
    env.state = np.zeros_like(env.state)
    env.steps = 0
    env.turn = 0
    # retornamos a observação consistente com essa configuração
    return env.get_representation_state(), {}


def play_single_game(env: HEX) -> None:
    """Roda uma partida humana vs adversário (humano = Player 1)."""
    obs, info = env.reset()
    # Garante tabuleiro vazio e humano começando
    obs, info = force_empty_start(env)

    print("\nNovo jogo iniciado!")
    print("Você é o Jogador 1 (X / vermelho). O adversário é o Jogador 2 (O / azul).")
    done = False

    while not done:
        env.render()
        print_valid_actions_grid(env)

        action = ask_human_action(env)
        if action is None:
            print("Saindo da partida.")
            return

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        if done:
            env.render()
            winner = info.get("winner", None)

            if terminated:
                if winner == 0:
                    print("\nFim de jogo: você venceu! 🎉")
                elif winner == 1:
                    print("\nFim de jogo: o adversário venceu.")
                else:
                    print("\nFim de jogo.")
            elif truncated:
                print("\nFim de jogo: tabuleiro cheio, nenhum vencedor claro.")
            break


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    device = choose_device(args.device)
    print(f"Usando dispositivo: {device}")

    # Construir adversário de acordo com o tipo
    if args.adversary_type == "random":
        print("Usando adversário aleatório (RandomAdversary).")
        adversary = RandomAdversary()
    else:
        # adversário NN
        print("Usando adversário de rede neural (NNAdversary com Attention_QNet).")
        model_path = resolve_model_path(args)
        model = build_model(args, device)

        if model_path is not None:
            try:
                load_model_weights(model, model_path, device)
            except Exception as e:
                print(f"Falha ao carregar modelo de {model_path}: {e}")
                print("Continuando com modelo não treinado.\n")
        else:
            print(
                "Nenhum modelo encontrado em models/attention_hex_adv.pth "
                "nem models/attention_hex.pth, e nenhum --model-path foi fornecido.\n"
                "Continuando com modelo não treinado.\n"
            )

        adversary = NNAdversary(
            model,
            n_best_actions=args.n_best_actions,
            device=device,
        )

    env = HEX(
        grid_size=args.grid_size,
        adversary=adversary,
        render_mode=args.render_mode,
        representation_mode="Matrix_Invertion",
        random_start=False,
    )

    try:
        while True:
            play_single_game(env)
            resp = input("\nJogar outra partida? [s/N] ").strip().lower()
            if resp not in {"s", "sim", "y", "yes"}:
                break
    finally:
        env.close()
        print("Encerrando.")


if __name__ == "__main__":
    # Garante que possamos importar `src` mesmo se o usuário rodar de outro lugar
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    main()
