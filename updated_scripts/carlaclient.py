import carla
from environment import ApplicationProperties

class CarlaClient( object ):

    def __init__( self, appProperties: ApplicationProperties ):
        self._appProperties = appProperties

    def inititaliseClient( self ):

        # The main purpose of the client object is to get or change the world, and apply commands
        self._carlaClient = carla.Client( self._appProperties.get_property_value( "carla.host" ), self._appProperties.get_property_value( "carla.port" ) )
        self._carlaClient.set_timeout( 2.0 )

        # A client can connect and retrieve the current world
        self._carlaWorld = self._carlaClient.get_world()
        # This module is in charge of every vehicle set to autopilot to recreate urban traffic.
        self._trafficManager = self._carlaClient.get_trafficmanager( self._appProperties.get_property_value( "carla.tmPort" ) )
        # Sets the minimum distance in meters that vehicles have to keep with the rest. The distance is in meters and will affect the minimum moving distance. It is computed from center to center of the vehicle objects.
        self._trafficManager.set_global_distance_to_leading_vehicle( 5.0 )

        # In this mode, vehicle's farther than a certain radius from the ego vehicle will have their physics disabled. Computation cost will be reduced by not calculating vehicle dynamics. Vehicles will be teleported.
        if self._appProperties.get_property_value( "carla.hybridPhysicsMode" ):
            self._trafficManager.set_hybrid_physics_mode( True )

        if self._appProperties.get_property_value( "carla.sync" ):
            CARLA_HZ = 20
            worldSettings = self._carlaWorld.get_settings() 
            # TM is designed to work in synchronous mode. Both the CARLA server and TM should be set to synchronous in order to function properly.
            self._trafficManager.set_synchronous_mode( True )  
            worldSettings.synchronous_mode = True
            #0.05 To run the simulation at a fixed time-step of 0.05 seconds apply the following settings. In this case, the simulator will take twenty steps (1/0.05) to recreate one second of the simulated world.
            worldSettings.fixed_delta_seconds = float( 1 / CARLA_HZ ) 
            self._carlaWorld.apply_settings( worldSettings )
