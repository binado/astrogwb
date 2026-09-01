# Regenerate the committed core mock-population fixture.
generate-mock-population-fixture:
    uv run --script packages/astrogwb/scripts/generate_mock_population_fixture.py \
        --population packages/astrogwb/tests/fixtures/mock_bns_population.yaml \
        --output packages/astrogwb/tests/fixtures/mock_bns_population.csv \
        --num-samples 1024 \
        --seed 41
