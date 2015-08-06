%--------------------------------------------------------------------------
%Script01b - Bouncing Ball
%Creating Simple Animation in MATLAB
%MATLAB Undercover
%zerocrossraptor.wordpress.com
%--------------------------------------------------------------------------
%This script m-file creates an endless animation of bouncing balls.
%--------------------------------------------------------------------------
 
%CodeStart-----------------------------------------------------------------
%Resetting MATLAB environment
    close all;
    clear all;
    clc;
%Declaring ball's initial condition
    initpos=50;     %Ball's initial vertical position
    initvel=0;      %Ball's initial vertical velocity
%Declaring environmental variable
    r_ball=5;       %Ball's radius
    gravity=10;     %Gravity's acceleration
    c_bounce=1;     %Bouncing's coefficient of elasticity
%Declaring animation timestep
    dt=0.0125;      %Animation timestep
%Drawing first frame
    rectangle('Position',[-r_ball,initpos,r_ball,r_ball],...
              'Curvature',[1,1],...
              'FaceColor','b');
    line([-5*r_ball,5*r_ball],...
         [0,0]);
%Executing animation
    pos=initpos-r_ball;             %Ball's current vertical position
    vel=initvel;                    %Ball's current vertical velocity
    while 1
        %Declaring time counter
        t_loopstart=tic();
        %Updating ball's condition
        pos=pos+(vel*dt);           %Ball's current vertical position
        vel=vel-(gravity*dt);       %Ball's current vertical velocity
        %Adjusting ball's velocity if the ball is hitting the floow
        if pos<0
            vel=-vel*c_bounce;      %Balls' current vertical velocity
        end
        %Clearing figure
        clf;
        %Drawing frame
        rectangle('Position',[-r_ball,pos,r_ball,r_ball],...
                  'Curvature',[1,1],...
                  'FaceColor','b');
%         line([-5*r_ball,5*r_ball],...
%              [0,0]);
        %Preserving axes
        axis([-8*r_ball,8*r_ball,0,initpos+2*r_ball]);
%         axis('equal');
%         axis('off');
        %Pausing animation
        el_time=toc(t_loopstart);
        disp(['Elapse time : ',num2str(el_time),' seconds']);
        pause(dt-el_time);
    end
%CodeEnd-------------------------------------------------------------------
