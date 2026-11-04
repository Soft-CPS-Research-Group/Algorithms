# Escolher um dataset Portugal

Os datasets pertencem ao Simulator. Identificar a família, a variante (por exemplo Core30 Normal), a versão do contrato e o ficheiro em `evaluation/`. Verificar o intervalo físico, a tarifa, a configuração de mercado e a presença de alterações de topologia no schema usado pelo job. A documentação de origem encontra-se em `docs/portugal_rec_native/README.md` no repositório Simulator.

O manifesto de experiências em [TI-MARL](../algorithms/ti_marl/experiments/portugal_evaluation_v2_manifest.json) tem uma versão própria. Confirmar que corresponde aos dados escolhidos antes de reutilizar splits ou regras de seleção. Guardar os hashes dos ficheiros efetivos e emparelhar controladores por janela e seed.

Para os comparadores comunitários, seguir o [protocolo Core30](baseline_comparison_core30.md).
