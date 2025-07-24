def left_aligned_pyramid(height):
    print("Left-Aligned Pyramid:")
    for i in range(1, height + 1):
        print("*" * i)
    print()


def right_aligned_pyramid(height):
    print("Right-Aligned Pyramid:")
    for i in range(1, height + 1):
        spaces = ' ' * (height - i)
        stars = '*' * i
        print(spaces + stars)
    print()


def centered_pyramid(height):
    print("Centered Pyramid:")
    for i in range(1, height + 1):
        spaces = ' ' * (height - i)
        stars = '*' * (2 * i - 1)
        print(spaces + stars + spaces)
    print()


# Main Program
if __name__ == "__main__":
    height = int(input("Enter the height of the pyramid: "))
    left_aligned_pyramid(height)
    right_aligned_pyramid(height)
    centered_pyramid(height)
