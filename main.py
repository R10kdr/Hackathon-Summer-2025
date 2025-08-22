import csv
import os


print("OUTPUT GENERATOR")
input_csv = 'data/test_set.csv'
output_csv = 'prediction/prediction.csv'

print(""" To compare two files, type "compare" and the file name.
      To run prediction, type "pseudoinverse", "ridge_regression", or "ridge_normal" and the output file (default: prediction/prediction.csv)
      """)
method = input("Enter method: ")
output_csv = input("Enter output CSV file path (default: prediction/prediction.csv): ") or output_csv

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
    # Run the rest of the code

    # Train model once
    coefficients, gene_names = train_with_pseudoinverse('data/train_set.csv')
    method = sys.argv[0]
    file = sys.argv[1] or output_csv
    with open(input_csv, newline='') as infile, open(output_csv, 'w', newline='') as outfile:
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
            match method:
                case "pseudoinverse":
                    predictions = predict_with_pseudoinverse(coefficients, gene_names, [gene1, gene2])
                case "ridge_regression":
                    predictions = train_with_regularization(coefficients, gene_names, [gene1, gene2])
                case "ridge_normal":
                    predictions = predict_expression(coefficients, gene_names, [gene1, gene2])
                case _:
                    print("Not a valid method")
                    print("Options: pseudoinverse, ridge_regression, ridge_normal")
            for gene, value in predictions.items():
                writer.writerow([gene, perturbation, value])
        print("PROCESS FINISHED")