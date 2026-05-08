function dZ = dLeakyRelu(dA, Z, alpha)

    dZ = dA;
    dZ(Z <= 0) = alpha * dZ(Z <= 0);

end