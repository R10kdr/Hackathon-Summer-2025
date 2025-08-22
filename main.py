import csv
import os
import sys


print("OUTPUT GENERATOR")
input_csv = 'data/test_set.csv'
output_csv = 'prediction/prediction.csv'

print(""" To compare two files, type "compare" and the file name.
      To run prediction, type "pseudoinverse", "ridge_regression", or "ridge_normal" and the output file (default: prediction/prediction.csv)
      """)
method = input("Enter method: ")
output_csv = input(
    "Enter output CSV file path (default: prediction/prediction.csv): ") or output_csv

# Ensure output directory exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

if method not in ["pseudoinverse", "ridge_regression", "ridge_normal", "compare"]:
    print("Unknown method")
    print("Options: pseudoinverse, ridge_regression, ridge_normal, compare")
    sys.exit(1)
if method == "compare":
    compare_csv(output_csv)
else:
    from pseudoinverse import *
    from ridge_regression import *
    from ridge_normal import *
    from rmsd import *
    # Train model(s) once
    if method == "pseudoinverse":
        coefficients, gene_names = train_with_pseudoinverse('data/train_set.csv')
    ridge_model = None
    ridge_gene_names = None
    ridge_n_genes = None
    if method == "ridge_normal":
        ridge_model, ridge_gene_names, ridge_n_genes = train_with_regularization('data/train_set.csv')
    with open(input_csv, newline='', encoding='utf-8') as infile, open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        # Write header
        writer.writerow(['gene', 'perturbation', 'expression'])
        for row in reader:
            print("Processing row:", row)
            if not row or len(row[0].split('+')) != 2:
                continue  # skip incomplete rows
            perturbation = row[0]
            gene1, gene2 = perturbation.split('+')
            if method == "pseudoinverse":
                predictions = predict_with_pseudoinverse(
                    coefficients, gene_names, [gene1, gene2])
            elif method == "ridge_normal":
                predictions = predict_with_regularization(
                    ridge_model, ridge_gene_names, ridge_n_genes, [gene1, gene2])
            elif method == "ridge_regression":
                # Assuming predict_expression expects (coefficients, gene_names, genes_to_perturb)
                predictions = predict_expression(coefficients, gene_names, [gene1, gene2])
            else:
                print("Not a valid method")
                print("Options: pseudoinverse, ridge_regression, ridge_normal")
            for gene, value in predictions.items():
                writer.writerow([gene, perturbation, value])
        print("PROCESS FINISHED")
