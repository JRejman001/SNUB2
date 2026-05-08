function A = leakyRelu(Z, alpha)
    A = max(Z, alpha * Z);
end