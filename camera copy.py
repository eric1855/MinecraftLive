import mouse_control
import time

print("Starting mouse control...")

# Scenario 1: Direct function calls
# Look Left (u)
mouse_control.look_left() 

time.sleep(1)

# Look Right (o)
mouse_control.look_right()

# Scenario 2: Passing the key variable
user_input = 'u' # Imagine this came from a key listener
mouse_control.handle_key(user_input)