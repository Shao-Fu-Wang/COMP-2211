def evaluate_polynomial(poly_string, x_value):
    # Split the polynomial string into individual terms
    terms = poly_string.split("+")

    # Evaluate each term and sum the results
    result = 0
    for term in terms:
        # Check if the term has a negative sign
        if "-" in term:
            # Split the term into coefficient and power parts
            parts = term.split("-")

            # Extract the coefficient and power
            coefficient = -int(parts[1].split("*")[0])
            power = int(parts[1].split("^")[1])
        else:
            # Split the term into coefficient and power parts
            parts = term.split("*")

            # Extract the coefficient and power
            coefficient = int(parts[0])
            power = int(parts[1].split("^")[1])

        # Evaluate the term for the given x_value
        result += coefficient * (x_value ** power)

    return result

# Example usage
poly_string = "-X^3-12*X^2+100"
x_value = 3
result = evaluate_polynomial(poly_string, x_value)
print("Evaluated Result:", result)