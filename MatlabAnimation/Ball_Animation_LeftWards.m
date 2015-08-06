%--------------------------------------------------------------------------
%This script m-file creates an endless animation of bouncing balls.
%--------------------------------------------------------------------------
 
%CodeStart-----------------------------------------------------------------
%Resetting MATLAB environment
    close all;
    clear all;
    clc;
%Declaring ball's initial condition
    r_ball=8;       %Ball's radius
    bound = 4*r_ball;
    initpos=bound;     %Ball's initial vertical position
    initvel=10;      %Ball's initial vertical velocity
    c_bounce=1;     %Bouncing's coefficient of elasticity
%Declaring animation timestep
    dt=0.0125;      %Animation timestep
%Drawing first frame
    rectangle('Position',[initpos,r_ball,r_ball,r_ball],...
              'Curvature',[1,1],...
              'FaceColor','b');
    line([-5*r_ball,5*r_ball],...
         [0,0]);
%Executing animation
    pos=initpos;             %Ball's current vertical position
    vel=initvel;             %Ball's current vertical velocity
    while 1
        %Declaring time counter
        t_loopstart=tic();
        %Updating ball's condition
        pos=pos-(vel*dt);           %Ball's current vertical position
%         vel=vel-(gravity*dt);       %Ball's current vertical velocity
        %Adjusting ball's velocity if the ball is hitting the floow
        if pos<0
            pos = initpos;      %Balls' current vertical velocity
        end
        %Clearing figure
        clf;
        %Drawing frame
        rectangle('Position',[pos,r_ball,r_ball,r_ball],...
                  'Curvature',[1,1],...
                  'FaceColor','b');
%         line([-5*r_ball,5*r_ball],...
%              [0,0]);
        %Preserving axes
        axis([0,bound,0,2*r_ball]);
        axis('equal');
        axis('off');
        %Pausing animation
        el_time=toc(t_loopstart);
        disp(['Elapse time : ',num2str(el_time),' seconds']);
        pause(dt-el_time);
    end
%CodeEnd-------------------------------------------------------------------
