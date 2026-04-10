import torch
from policy import TransformerPolicy
# --- Utilities ---
def max_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()
def l2_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.norm(a - b).item()
def format_vec(v: torch.Tensor, k: int = 3) -> str:
    v = v.detach().cpu().flatten()
    k = min(k, v.numel())
    vals = ", ".join([f"{x:.4f}" for x in v[:k]])
    return f"[{vals}{', ...' if v.numel() > k else ''}]"
def print_case_header(name: str) -> None:
    print(f"\n{name}")
    print("-" * len(name))
def make_inputs(num_cas: int, num_sros: int, embedding_dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator()
    g.manual_seed(seed)

    ca_embeddings = torch.randn(num_cas, embedding_dim, generator=g)
    sro_embeddings = torch.randn(num_sros, embedding_dim, generator=g)
    nfc_embedding = torch.randn(embedding_dim, generator=g)
    return ca_embeddings, sro_embeddings, nfc_embedding

# --- Test 1: Variable CA cardinality ---
def test_variable_ca_count(policy: TransformerPolicy, embedding_dim: int) -> None:
    print_case_header("Test 1, variable CA cardinality (R1)")

    fixed_sros = 5
    for num_cas in [3, 7]:
        ca, sro, nfc = make_inputs(num_cas, fixed_sros, embedding_dim, seed=1000 + num_cas)
        outputs = policy(ca, sro, nfc)

        observed_cas = ca.shape[0]
        observed_sros = sro.shape[0]
        observed_outputs = outputs.shape[0]

        print(
            f"Input tokens, CAs={observed_cas}, SROs={observed_sros}, NFC=1 "
            f"-> Output vectors (CA actions)={observed_outputs} (expected {observed_cas})"
        )

# --- Test 2: Variable SRO cardinality ---
def test_variable_sro_count(policy: TransformerPolicy, embedding_dim: int) -> None:
    print_case_header("Test 2, output cardinality tied to CAs (R2)")

    fixed_cas = 4
    for num_sros in [0, 9]:
        ca, sro, nfc = make_inputs(fixed_cas, num_sros, embedding_dim, seed=2000 + num_sros)
        outputs = policy(ca, sro, nfc)

        print(
            f"Input tokens, CAs={ca.shape[0]}, SROs={sro.shape[0]}, NFC=1 "
            f"-> Output vectors (CA actions)={outputs.shape[0]} (expected {ca.shape[0]})"
        )

# --- Test 3: Contextual sensitivity ---
def test_contextual_sensitivity(policy: TransformerPolicy, embedding_dim: int) -> None:
    print_case_header("Test 3, contextual influence (R3)")

    num_cas = 4
    num_sros = 6
    ca, sro, nfc = make_inputs(num_cas, num_sros, embedding_dim, seed=3000)

    base = policy(ca, sro, nfc)

    ca_idx = 0
    print(f"Baseline, example CA[{ca_idx}] output: {format_vec(base[ca_idx])}\n")

    delta = 0.5

    # Perturb NFC only
    nfc_perturbed = nfc.clone()
    nfc_perturbed[0] += delta
    out_nfc = policy(ca, sro, nfc_perturbed)

    print(f"After NFC perturbation, CA[{ca_idx}] output: {format_vec(out_nfc[ca_idx])}")
    print(
        f"NFC perturbation deltas, max abs={max_abs_delta(base, out_nfc):.6f}, L2={l2_delta(base, out_nfc):.6f}\n"
    )

    # Perturb one SRO token only
    sro_perturbed = sro.clone()
    sro_perturbed[0, 0] += delta
    out_sro = policy(ca, sro_perturbed, nfc)

    print(f"After SRO[0] perturbation, CA[{ca_idx}] output: {format_vec(out_sro[ca_idx])}")
    print(
        f"SRO perturbation deltas, max abs={max_abs_delta(base, out_sro):.6f}, L2={l2_delta(base, out_sro):.6f}\n"
    )

    # Perturb CA[1] and observe effect on CA[0] (cross-token coupling)
    ca_perturbed = ca.clone()
    ca_perturbed[1, 0] += delta
    out_ca = policy(ca_perturbed, sro, nfc)

    print(f"After perturbing CA[1], CA[{ca_idx}] output: {format_vec(out_ca[ca_idx])}")
    print(
        f"CA perturbation deltas, max abs={max_abs_delta(base, out_ca):.6f}, L2={l2_delta(base, out_ca):.6f}\n"
    )

# --- Main Routine ---
def main() -> None:
    torch.manual_seed(42)
    embedding_dim = 64
    policy = TransformerPolicy( embedding_dim=embedding_dim, num_attention_heads=4,num_encoder_layers=2, 
        feedforward_dim=128, action_dim=3)
    policy.eval()

    test_variable_ca_count(policy, embedding_dim)
    test_variable_sro_count(policy, embedding_dim)
    test_contextual_sensitivity(policy, embedding_dim)

if __name__ == "__main__":
    main()