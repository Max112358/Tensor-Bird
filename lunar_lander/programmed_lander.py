from programmable_lander import ProgrammableLunarLander

class MyLunarLander(ProgrammableLunarLander):
    def compute_control(self):
        # Get state from input handler
        state = self.input_handler.get_state(self.lander, self.terrain)
        
        # Extract normalized values
        velocity_x = state[0]
        velocity_y = state[1]  # Normalized by safe landing velocity
        angle = state[2]       # Normalized by safe landing angle -1 for angled left, 1 for angled right
        angular_vel = state[3]
        dist_x = state[4]      # Normalized [0,1]
        dist_y = state[5]      # Normalized [0,1]
        
        correct_angle = dist_x
        
        print("angle: " + str(angle) )
        print("correct_angle: " + str(correct_angle) )
        #print("correct angle:" + str(correct_angle) )
        
        # Example control logic using normalized inputs
        
        
        #slow decent
        if velocity_y - dist_y   > 0.3:
            return 2
        
        
        if angle + correct_angle - angular_vel  > 0:
            return 1
        
        if angle + correct_angle - angular_vel < 0:
            return 3
        
        
        '''
        #breaking thruster
        if velocity_y > 0.8 and dist_y < 0.40:
            return 2
        
        #move right thruster
        if dist_x > 0 and angle > 0.2 and velocity_y > 0:
            return 2
        
        #move left thruster
        if dist_x < 0 and angle < -0.2 and velocity_y > 0:
            return 2
        
        if angle < correct_angle:
            if abs(angular_vel) < 1.8:
                return 3
        
        if angle > correct_angle:
            if abs(angular_vel) < 1.8:
                return 1
        '''
        
        
        return 0
        

if __name__ == "__main__":
    game = MyLunarLander()
    try:
        game.run()
    finally:
        game.close()