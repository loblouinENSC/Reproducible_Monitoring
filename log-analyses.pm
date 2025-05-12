# User-defined operators & Global lets

sub sensor_failure() {
  my ($s,$T) = @_;
  sub{[&or, [&ge($T), $s], [&ge($T), [&not, $s]]]}->();
}
sub outing_period() {
  my ($no_activity,$T) = @_;
  sub{&holds($no_activity,[&gt($T), &between(&dn("ContactS_E"),&up("ContactS_E"))])}->();
}
sub platform_failure() {
  my ($no_activity,$T) = @_;
  sub{[&and, [&gt($T), $no_activity], [&not, &outing_period($no_activity,$T)]]}->();
}
sub platform_failure_1day() {
  my ($no_activity) = @_;
  sub{&platform_failure($no_activity,86400000)}->();
}
sub sleep_segment() {
  my ($any_motion_up) = @_;
  sub{&holds([&not, $any_motion_up],[&or, [&le(36000000), [&ge(1800000), "MotionD_B"]], [&le(36000000), [&ge(1800000), [&not, "MotionD_B"]]]])}->();
}
sub sleep_segment_ext() {
  my ($no_presence,$night) = @_;
  sub{[&le(36000000), [&ge(1800000), [&or, &until([&and, $no_presence, $night],"MotionD_B"), &since([&and, $no_presence, $night],"MotionD_B")]]]}->();
}
my $any_emeter_up=&any_up("EMeter_Cofeemaker","EMeter_L"); my $any_contact_sw=&any_sw("ContactS_Cupboard","ContactS_E","ContactS_Fridge"); my $any_motion_up_but_bed=&any_up("MotionD_B","MotionD_E","MotionD_K","MotionD_L","MotionD_S","MotionD_T"); my $any_motion_up=[&or, $any_motion_up_but_bed, &up("MotionD_B")]; my $any_activity=[&or, $any_motion_up, [&or, $any_emeter_up, $any_contact_sw]]; my $no_activity=[&not, $any_activity]; my $any_motion=&any("MotionD_B","MotionD_E","MotionD_K","MotionD_L","MotionD_S","MotionD_T"); my $any_presence=[&or, $any_motion, [&or, $any_emeter_up, $any_contact_sw]]; my $no_presence=[&not, $any_presence]; my $outing_period_10min=&outing_period($no_activity,600000); my $night=[&or, &slot_2017(79200000,28800000), &slot_2018(79200000,28800000)]; my $sleep_segment=&sleep_segment($any_motion_up_but_bed); my $sleep=&ex($night,[&or, $sleep_segment, &during([&le(900000), [&not, $sleep_segment]],$night)]); my $quiet_night_segment=[&le(36000000), [&ge(1800000), [&and, &ex($night,$no_activity), [&not, $sleep_segment]]]]; my $quiet_sleep_segment=[&or, $sleep_segment, $quiet_night_segment]; my $sleep_quiet=&ex($night,[&or, $quiet_sleep_segment, &during([&le(900000), [&not, $quiet_sleep_segment]],$night)]); 
# Result structure
(

# Requires (uses)
{
},

# Provides (defs)
{
outing_period => [1, 1, ""],
platform_failure => [1, 1, ""],
platform_failure_1day => [0, 1, ""],
sensor_failure => [1, 1, ""],
sleep_segment => [0, 1, ""],
sleep_segment_ext => [0, 2, ""],
},

# User-defined contexts
[
"door_failure_1week",
sub{my $nodoor=&sensor_failure("ContactS_E",604800000); $nodoor}->(),
"platform_failure_1day",
sub{&platform_failure_1day($no_activity)}->(),
"outing",
sub{$outing_period_10min}->(),
"toilet",
sub{[&or, [&le(1200000), "MotionD_T"], &holds([&not, $any_motion_up],[&le(180000), [&not, "MotionD_T"]])]}->(),
"toilet_failure",
sub{my $nout=[&not, $outing_period_10min]; [&or, [&ge(86400000), [&and, [&not, "MotionD_T"], $nout]], [&ge(86400000), [&and, "MotionD_T", $nout]]]}->(),
"bed_failure",
sub{my $nout=[&not, $outing_period_10min]; [&or, [&ge(86400000), [&and, [&not, "MotionD_B"], $nout]], [&ge(86400000), [&and, "MotionD_B", $nout]]]}->(),
"sleep",
sub{$sleep}->(),
"quiet_night_segment",
sub{$quiet_night_segment}->(),
"sleep_quiet",
sub{$sleep_quiet}->(),
]
)
