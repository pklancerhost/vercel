import random

# Define available interests
INTEREST_B1G1 = "Buy 1 Get 1"
INTEREST_DISCOUNT = "Discount"
INTEREST_BOTH = "Both"

# 1. Generate the Dictionary
# Target counts: 5x Buy 1 Get 1, 7x Discount, 3x Both (Total 15 users)
user_interests = {}

# Add 5 users interested in "Buy 1 Get 1"
for i in range(1, 6):
    user_interests[f"user_{i}"] = INTEREST_B1G1

# Add 7 users interested in "Discount"
for i in range(6, 13):
    user_interests[f"user_{i}"] = INTEREST_DISCOUNT

# Add 3 users interested in "Both"
for i in range(13, 16):
    user_interests[f"user_{i}"] = INTEREST_BOTH


print("--- Generated User Dictionary ---")
print(user_interests)
print("\n" + "=" * 40 + "\n")


# 2. Process Users using Loops and Simple Conditions
b1g1_count = 0
discount_count = 0
both_count = 0

print("--- Categorized User Offers ---")

# Loop through the dictionary items (Username and Interest)
for username, interest in user_interests.items():
    # Simple conditional logic to evaluate user interest
    if interest == INTEREST_B1G1:
        b1g1_count += 1
        print(f"{username}: Eligible for Buy 1 Get 1 Free Promo")

    elif interest == INTEREST_DISCOUNT:
        discount_count += 1
        print(f"{username}: Eligible for Percentage Discount Code")

    elif interest == INTEREST_BOTH:
        both_count += 1
        print(f"{username}: Eligible for ALL Special Offers (Both B1G1 and Discount)")

    else:
        print(f"{username}: No matching interest found")


# 3. Display Summary Totals
print("\n" + "=" * 40 + "\n")
print("--- Interest Summary ---")
print(f"Total Users interested in 'Buy 1 Get 1': {b1g1_count}")
print(f"Total Users interested in 'Discount'    : {discount_count}")
print(f"Total Users interested in 'Both'        : {both_count}")
print(f"Total Processed Users                  : {len(user_interests)}")