# def implies(p, q):
#     """Helper function for material implication (P -> Q is equivalent to not P or Q)."""
#     return (not p) or q
# def format_bool(b):
#     """Converts Python boolean (True/False) to 'T' or 'F'."""
#     return "T" if b else "F"
# # Print header
# print(f"{'P':<5} | {'Q':<5} | {'P->Q':<7} | {'Result':<6}")
# print("-" * 32)
# # Evaluate all truth value combinations for P and Q
# for P in [False, True]:
#     for Q in [False, True]:
#         p_implies_q = implies(P, Q)
#         # (P AND (P -> Q))
#         left_side = P and p_implies_q
#         # (P AND (P -> Q)) -> Q
#         result = implies(left_side, Q)
#         print(f"{format_bool(P):<5} | {format_bool(Q):<5} | {format_bool(p_implies_q):<7} | {format_bool(result):<6}")
parent = {('Tom','Bob'), ('Bob','Ann'),
          ('Bob','Pat')}
people = {p for pair in parent for p in pair} #comprehension to get all unique people in the parent set
 
def grandparent(x, z):
    return any((x, y) in parent and
               (y, z) in parent
               for y in people)
 
print(grandparent('Tom','Bob'))  # True
print(grandparent('Tom','Pat'))  # False
