UPDATE vendas
SET Produto = CASE
    WHEN Produto = 11 THEN 'Gel de Limpeza Facial'
    WHEN Produto = 12 THEN 'Gel de Limpeza Corporal'
    WHEN Produto = 13 THEN 'Gel de Limpeza Antioleosidade'
    WHEN Produto = 21 THEN 'Hidratante Facial'
    WHEN Produto = 22 THEN 'Hidratante Corporal'
    WHEN Produto = 23 THEN 'Hidratante Antioleosidade'
    WHEN Produto = 31 THEN 'Protetor Solar Facial'
    WHEN Produto = 32 THEN 'Protetor Solar Corporal'
    WHEN Produto = 33 THEN 'Protetor Solar Antioleosidade'
END;


SELECT
    Produto, Categoria,
    SUM(Quantidade * "Preço Unitario ($)") AS Total_Vendas
FROM
    vendas
GROUP BY
    Produto
ORDER BY
    Total_Vendas DESC;


SELECT
    Produto,
    SUM(Quantidade * "Preço Unitario ($)") AS Total_Vendas
FROM
    vendas
WHERE
    Data = 6
GROUP BY
    Produto
ORDER BY
    Total_Vendas ASC;
