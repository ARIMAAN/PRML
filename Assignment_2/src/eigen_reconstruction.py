import numpy as np
from evd import matrix_inverse


def reconstruct_image(V, D, K):

    # Get eigenvalues from diagonal of D
    eigenvalues = np.diag(D)

    n = len(eigenvalues)

    # Sort eigenvalues according to magnitude
    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]

    # Initially select top K
    selected_indices = list(sorted_indices[:K])

    # ---------------------------------------------------------
    # Make sure complex conjugate pairs are included
    # ---------------------------------------------------------

    for idx in selected_indices.copy():

        eigenvalue = eigenvalues[idx]

        # Only check complex eigenvalues
        if not np.isclose(eigenvalue.imag, 0):

            conjugate = np.conj(eigenvalue)

            # Find conjugate index
            for j in range(n):

                if np.isclose(eigenvalues[j], conjugate):

                    conjugate_index = j

                    # If conjugate isn't already selected
                    if conjugate_index not in selected_indices:

                        # Remove the last selected eigenvalue
                        selected_indices.pop()

                        # Add conjugate
                        selected_indices.append(conjugate_index)

                    break

    # ---------------------------------------------------------
    # Create D_k
    # All values are initially ZERO
    # ---------------------------------------------------------

    D_k = np.zeros_like(D, dtype=complex)

    # Put only selected eigenvalues into D_k
    for idx in selected_indices:
        D_k[idx, idx] = eigenvalues[idx]

    # ---------------------------------------------------------
    # Calculate V inverse
    # ---------------------------------------------------------

    V_inverse = matrix_inverse(V)

    # ---------------------------------------------------------
    # Reconstruction
    # X = V D_k V^-1
    # ---------------------------------------------------------

    X = V @ D_k @ V_inverse

    # Remove tiny numerical imaginary components
    if np.allclose(X.imag, 0):
        X = X.real

    return X